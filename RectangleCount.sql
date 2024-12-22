SELECT count(*)
FROM Edge AS E1, Edge AS E2, Edge AS E3, E4
WHERE E1.dst = E2.src 
AND E2.dst = E3.src 
AND E3.dst = E4.src
AND E4.dst = E1.src
AND E1.src < min(E2.src, E3.src, E4.src)

