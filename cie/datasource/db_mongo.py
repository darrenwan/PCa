import pymongo
from sshtunnel import SSHTunnelForwarder
from cie.common.settings import MongoDB_CONFIG

#cur.execute("SELECT Host,User FROM user")

class MongoClient(object):
    def __init__(self,config):
        self.config = config

        if config["local"]:
            server = SSHTunnelForwarder((config["ssh_host"], config["ssh_port"]),
                                        ssh_username=config["ssh_user"],
                                        # ssh_password=config["ssh_password"],
                                        ssh_private_key=config["ssh_private_key"],
                                        remote_bind_address=(config["host"], config["port"])
                                        )
            server.start()
            self.client = pymongo.MongoClient('localhost', server.local_bind_port)
            self.db = self.__set_database(config)
        else:
            self.client = pymongo.MongoClient(config["host"], config["port"])
            self.db = self.__set_database(config)


    def __set_database(self,config):
        self.client[config["dbname"]].authenticate(
            config["user"],
            config["password"],
            mechanism='SCRAM-SHA-1')
        return self.client[config["dbname"]]

    def close(self):
        self._cursor.close()
        self._conn.close()

    def merge_documents(self, documents):
        from collections import defaultdict
        records = defaultdict()

            
    def execute(self, sql, database='dpp-test', collection='candidate'):
        print('将执行的sql语句是: db["%s"]["%s"].%s' % (database, collection, sql))
        
        self._conn = self.client[database]
        self._conn.authenticate(self.user, self.password, mechanism="SCRAM-SHA-1")
        
        try:
            documents = eval('self._conn[collection].' + sql)            
            return documents
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':
    mgcl = MongoClient(MongoDB_CONFIG)
    mgcl.db
