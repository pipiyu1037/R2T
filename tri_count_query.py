import psycopg2
import numpy as np
from R2T import R2T

def get_db_connection():
    return psycopg2.connect(
        dbname="Deezer",
        user="postgres",
        password="Yujinqi.2002",
        host="localhost",
        port="5432"
    )
    
query = """
SELECT count(*)
FROM Edge AS E1, Edge AS E2, Edge AS E3
WHERE E1.dst = E2.src 
AND E2.dst = E3.src
AND E3.dst = E1.src
AND E1.dst > E1.src
AND E2.dst > E2.src
"""

rewritten_query = """
SELECT Node.id, 1
FROM Node, Edge AS E1, Edge AS E2, Edge AS E3
WHERE Node.id = E1.src
AND E1.dst = E2.src 
AND E2.dst = E3.src
AND E3.dst = E1.src
AND E1.dst > E1.src
AND E2.dst > E2.src
"""

if __name__ == "__main__":
    conn = get_db_connection()
    r2t = R2T(gs_Q=2048)
    result = r2t.query(conn, rewritten_query)
    true_Res = r2t.aux_query(conn, query)
    print("True Q:" + str(true_Res[0][0]))
    conn.close()
    print("R2T Q:" + str(result))