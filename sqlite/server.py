from sqlite_rx.server import SQLiteServer

def main():

    # database is a path-like object giving the pathname 
    # of the database file to be opened. 
    
    # You can use ":memory:" to open a database connection to a database 
    # that resides in RAM instead of on disk

    server = SQLiteServer(database="recsys05.db",
                          bind_address="tcp://0.0.0.0:30157")
    server.start()
    server.join()

if __name__ == '__main__':
    main()
