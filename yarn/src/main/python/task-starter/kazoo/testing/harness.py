"""Kazoo testing harnesses"""

import logging
import os
import uuid
import unittest

from kazoo import python2atexit as atexit

from kazoo.client import KazooClient
from kazoo.exceptions import KazooException, NotEmptyError
from kazoo.protocol.states import (
    KazooState
)
from kazoo.testing.common import ZookeeperCluster
from kazoo.protocol.connection import _CONNECTION_DROP, _SESSION_EXPIRED

log = logging.getLogger(__name__)

CLUSTER = None


def get_global_cluster():
    global CLUSTER
    if CLUSTER is None:
        ZK_HOME = os.environ.get("ZOOKEEPER_PATH")
        ZK_CLASSPATH = os.environ.get("ZOOKEEPER_CLASSPATH")
        ZK_PORT_OFFSET = int(os.environ.get("ZOOKEEPER_PORT_OFFSET", 20000))

        assert ZK_HOME or ZK_CLASSPATH, (
            "Either ZOOKEEPER_PATH or ZOOKEEPER_CLASSPATH environment "
            "variable must be defined.\n"
            "For deb package installations this is /usr/share/java")

        CLUSTER = ZookeeperCluster(
            install_path=ZK_HOME,
            classpath=ZK_CLASSPATH,
            port_offset=ZK_PORT_OFFSET,
        )
        atexit.register(lambda cluster: cluster.terminate(), CLUSTER)
    return CLUSTER


class KazooTestHarness(unittest.TestCase):
    """Harness for testing code that uses Kazoo

    This object can be used directly or as a mixin. It supports starting
    and stopping a complete ZooKeeper cluster locally and provides an
    API for simulating errors and expiring sessions.

    Example::

        class MyTestCase(KazooTestHarness):
            def setUp(self):
                self.setup_zookeeper()

                # additional test setup

            def tearDown(self):
                self.teardown_zookeeper()

            def test_something(self):
                something_that_needs_a_kazoo_client(self.client)

            def test_something_else(self):
                something_that_needs_zk_servers(self.servers)

    """

    def __init__(self, *args, **kw):
        super(KazooTestHarness, self).__init__(*args, **kw)
        self.client = None
        self._clients = []

    @property
    def cluster(self):
        return get_global_cluster()

    @property
    def servers(self):
        return ",".join([s.address for s in self.cluster])

    def _get_nonchroot_client(self):
        c = KazooClient(self.servers)
        self._clients.append(c)
        return c

    def _get_client(self, **kwargs):
        c = KazooClient(self.hosts, **kwargs)
        self._clients.append(c)
        return c

    def lose_connection(self, event_factory):
        """Force client to lose connection with server"""
        self.__break_connection(_CONNECTION_DROP, KazooState.SUSPENDED, event_factory)

    def expire_session(self, event_factory):
        """Force ZK to expire a client session"""
        self.__break_connection(_SESSION_EXPIRED, KazooState.LOST, event_factory)

    def setup_zookeeper(self, **client_options):
        """Create a ZK cluster and chrooted :class:`KazooClient`

        The cluster will only be created on the first invocation and won't be
        fully torn down until exit.
        """
        do_start = False
        for s in self.cluster:
            if not s.running:
                do_start = True
        if do_start:
            self.cluster.start()
        namespace = "/kazootests" + uuid.uuid4().hex
        self.hosts = self.servers + namespace
        if 'timeout' not in client_options:
            client_options['timeout'] = 0.8
        self.client = self._get_client(**client_options)
        self.client.start()
        self.client.ensure_path("/")

    def teardown_zookeeper(self):
        """Reset and cleanup the zookeeper cluster that was started."""
        while self._clients:
            c = self._clients.pop()
            try:
                c.stop()
            except KazooException:
                log.exception("Failed stopping client %s", c)
            finally:
                c.close()
        self.client = None

    def __break_connection(self, break_event, expected_state, event_factory):
        """Break ZooKeeper connection using the specified event."""

        lost = event_factory()
        safe = event_factory()

        def watch_loss(state):
            if state == expected_state:
                lost.set()
            elif lost.is_set() and state == KazooState.CONNECTED:
                safe.set()
                return True

        self.client.add_listener(watch_loss)
        self.client._call(break_event, None)

        lost.wait(5)
        if not lost.isSet():
            raise Exception("Failed to get notified of broken connection.")

        safe.wait(15)
        if not safe.isSet():
            raise Exception("Failed to see client reconnect.")

        self.client.retry(self.client.get_async, '/')


class KazooTestCase(KazooTestHarness):
    def setUp(self):
        self.setup_zookeeper()

    def tearDown(self):
        self.teardown_zookeeper()
