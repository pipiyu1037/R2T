from R2T import R2T
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        dbname="Deezer",
        user="postgres",
        password="Yujinqi.2002",
        host="localhost",
        port="5432"
    )

query = """
SELECT count(DISTINCT Edge.dst) 
FROM Node, Edge
WHERE Node.ID = Edge.src
"""

proj_query = """
SELECT DISTINCT Edge.dst, 1
FROM Node, Edge
WHERE Node.ID = Edge.src
"""

rewritten_query = """
SELECT Edge.dst, 1
FROM Node, Edge
WHERE Node.ID = Edge.src
"""

EPSILON = 0.8
BETA = 0.1
GSQ = 16384*2

def main():
    conn = get_db_connection()
    r2t = R2T(gs_Q=GSQ)
    result = r2t.query(conn, rewritten_query, proj_query)
    conn.close()
    print(result)
    

if __name__ == "__main__":
    main()