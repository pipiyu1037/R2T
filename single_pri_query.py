import psycopg2
import numpy as np
from R2T import R2T

def get_db_connection():
    return psycopg2.connect(
        dbname="tpch1g",
        user="postgres",
        password="Yujinqi.2002",
        host="localhost",
        port="5432"
    )
    
single_private_query = """
SELECT customer.c_custkey, l_extendedprice * (1 - l_discount) / 1000
FROM supplier, lineitem, orders, customer
WHERE supplier.s_suppkey = lineitem.l_suppkey AND lineitem.l_orderkey = orders.o_orderkey
AND orders.o_custkey = customer.c_custkey AND orders.o_orderdate >= '1997-07-01'
"""

query = """
SELECT SUM(l_extendedprice * (1 - l_discount) / 1000)
FROM supplier, lineitem, orders, customer
WHERE supplier.s_suppkey = lineitem.l_suppkey AND lineitem.l_orderkey = orders.o_orderkey
AND orders.o_custkey = customer.c_custkey AND orders.o_orderdate >= '1997-07-01'
"""
if __name__ == "__main__":
    conn = get_db_connection()
    r2t = R2T(gs_Q=2048)
    result = r2t.query(conn, single_private_query)
    true_Q = r2t.aux_query(conn, query)
    conn.close()
    print("True Q:" + str(float(true_Q[0][0])))
    print("R2T Q:" + str(result))