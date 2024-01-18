# client.py

from sqlite_rx.client import SQLiteClient

client = SQLiteClient(connect_address="tcp://10.28.224.124:30157")

with client:
  query = "CREATE TABLE stocks (date text, trans text, symbol text, qty real, price real)"
  result = client.execute(query)
