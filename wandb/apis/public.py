import logging
import requests
import time
import sys
import os
import json
import re
import six
from gql import Client, gql
from gql.client import RetryError
from gql.transport.requests import RequestsHTTPTransport

import wandb
from wandb import Error, __version__
from wandb import util
from wandb.summary import HTTPSummary
from wandb.apis import normalize_exceptions

logger = logging.getLogger(__name__)

RUN_FRAGMENT = '''fragment RunFragment on Run {
    id
    tags
    name
    state
    config
    readOnly
    createdAt
    heartbeatAt
    description
    systemMetrics
    summaryMetrics
}'''


class Api(object):
    """W&B Public API

    Args:
        setting_overrides(:obj:`dict`, optional): You can set defaults such as
        username, project, and run here as well as which api server to use.
    """

    HTTP_TIMEOUT = 10

    def __init__(self, overrides={}):
        self.settings = {
            'username': None,
            'project': None,
            'run': "latest",
            'base_url': "https://api.wandb.ai"
        }
        self._runs = {}
        self.settings.update(overrides)

    @property
    def client(self):
        return Client(
            transport=RequestsHTTPTransport(
                headers={'User-Agent': self.user_agent},
                use_json=True,
                # this timeout won't apply when the DNS lookup fails. in that case, it will be 60s
                # https://bugs.python.org/issue22889
                timeout=self.HTTP_TIMEOUT,
                auth=("api", self.api_key),
                url='%s/graphql' % self.settings['base_url']
            )
        )

    @property
    def user_agent(self):
        return 'W&B Public Client %s' % __version__

    @property
    def api_key(self):
        auth = requests.utils.get_netrc_auth(self.settings['base_url'])
        key = None
        if auth:
            key = auth[-1]
        # Environment should take precedence
        if os.getenv("WANDB_API_KEY"):
            key = os.environ["WANDB_API_KEY"]
        return key

    def _parse_path(self, path):
        run = self.settings['run']
        project = self.settings['project']
        username = self.settings['username']
        parts = path.replace("/runs/", "/").split("/")
        if parts[-1]:
            run = parts[-1]
        if len(parts) > 1:
            project = parts[1]
            if username and run == project:
                project = parts[0]
            else:
                username = parts[0]
        return (username, project, run)

    def runs(self, path="", filters={}):
        """Return a set of runs from a project that match the filters provided.
        You can filter by config.*, summary.*, state, username, createdAt, etc.

        The filters use the same query language as MongoDB:

        https://docs.mongodb.com/manual/reference/operator/query
        """
        username, project, run = self._parse_path(path)
        if not self._runs.get(path):
            self._runs[path + str(filters)] = Runs(self.client,
                                                   username, project, filters)
        return self._runs[path + str(filters)]

    @normalize_exceptions
    def run(self, path=""):
        """Returns a run by parsing path in the form username/project/run, if
        defaults were set on the Api, only overrides what's passed.  I.E. you can just pass
        run_id if you set username and project on the Api"""
        username, project, run = self._parse_path(path)
        if not self._runs.get(path):
            self._runs[path] = Run(self.client, username, project, run)
        return self._runs[path]


class Runs(object):
    QUERY = gql('''
        query Runs($project: String!, $entity: String!, $cursor: String, $filters: JSONString) {
            project(name: $project, entityName: $entity) {
                runCount(filters: $filters)
                readOnly
                runs(filters: $filters, after: $cursor) {
                    edges {
                        node {
                            ...RunFragment
                        }
                        cursor
                    }
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                }
            }
        }
        %s
        ''' % RUN_FRAGMENT)

    def __init__(self, client, username, project, filters={}):
        self.client = client
        self.username = username
        self.project = project
        self.filters = filters
        self.runs = []
        self.length = None
        self.index = -1
        self.cursor = None
        self.more = True

    def __iter__(self):
        return self

    def __len__(self):
        if self.length is None:
            self._load_page()
        return self.length

    def _load_page(self):
        if not self.more:
            return False
        res = self.client.execute(self.QUERY, variable_values={
            'project': self.project, 'entity': self.username,
            'filters': json.dumps(self.filters), 'cursor': self.cursor})
        self.length = res['project']['runCount']
        self.more = res['project']['runs']['pageInfo']['hasNextPage']
        if self.length > 0:
            self.cursor = res['project']['runs']['edges'][-1]['cursor']
        self.runs.extend([Run(self.client, self.username, self.project, r["node"]["name"], r["node"])
                          for r in res['project']['runs']['edges']])
        return True

    def __getitem__(self, index):
        loaded = True
        while loaded and index > len(self.runs):
            loaded = self._load_page()
        return self.runs[index]

    def __next__(self):
        self.index += 1
        if len(self.runs) <= self.index:
            if not self._load_page():
                raise StopIteration
        return self.runs[self.index]

    next = __next__

    def __repr__(self):
        return "<Runs {}/{} ({})>".format(self.username, self.project, len(self))


class Run(object):
    """A single run associated with a user and project"""

    def __init__(self, client, username, project, name, attrs={}):
        self.client = client
        self.username = username
        self.project = project
        self.name = name
        self._summary = None
        self._attrs = attrs
        self.load()

    def load(self, force=False):
        query = gql('''
        query Run($project: String!, $entity: String!, $name: String!) {
            project(name: $project, entityName: $entity) {
                run(name: $name) {
                    ...RunFragment
                }
            }
        }
        %s
        ''' % RUN_FRAGMENT)
        if force or not self._attrs:
            response = self._exec(query)
            self._attrs = response['project']['run']
        summary_metrics = json.loads(
            self._attrs['summaryMetrics'])
        # TODO: convert arrays into nparrays
        self._attrs['summaryMetrics'] = summary_metrics
        self._attrs['systemMetrics'] = json.loads(self._attrs['systemMetrics'])
        config = {}
        for key, value in six.iteritems(json.loads(self._attrs['config'])):
            if isinstance(value, dict) and value.get("value"):
                config[key] = value["value"]
            else:
                config[key] = value
        self._attrs['config'] = config
        return self._attrs

    def snake_to_camel(self, string):
        camel = "".join([i.title() for i in string.split("_")])
        return camel[0].lower() + camel[1:]

    def __getattr__(self, name):
        key = self.snake_to_camel(name)
        if key in self._attrs.keys():
            return self._attrs[key]
        elif name in self._attrs.keys():
            return self._attrs[name]
        else:
            raise AttributeError("'Run' object has no attribute '%s'" % name)

    def _exec(self, query, **kwargs):
        """Execute a query against the cloud backend"""
        variables = {'entity': self.username,
                     'project': self.project, 'name': self.name}
        variables.update(kwargs)
        return self.client.execute(query, variable_values=variables)

    @normalize_exceptions
    def history(self, samples=500, pandas=True, stream="default"):
        """Return history metrics for a run

        Args:
            samples (int, optional): The number of samples to return
            pandas (bool, optional): Return a pandas dataframe
            stream (str, optional): "default" for metrics, "system" for machine metrics
        """
        node = "history" if stream == "default" else "events"
        query = gql('''
        query Run($project: String!, $entity: String!, $name: String!, $samples: Int!) {
            project(name: $project, entityName: $entity) {
                run(name: $name) { %s(samples: $samples) }
            }
        }
        ''' % node)

        response = self._exec(query, samples=samples)
        lines = [json.loads(line)
                 for line in response['project']['run'][node]]
        if pandas:
            try:
                import pandas
                lines = pandas.DataFrame.from_records(lines)
            except ImportError:
                print("Unable to load pandas, call history with pandas=False")
        return lines

    @property
    def summary(self):
        if self._summary is None:
            self._summary = HTTPSummary(
                self.client, self.id, self.summary_metrics)
        return self._summary

    def __repr__(self):
        return "<Run {}/{}/{} ({})>".format(self.username, self.project, self.name, self.state)
