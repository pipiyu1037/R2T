SELECT count(*)
FROM Edge AS E1, Edge AS E2, Edge AS E3
WHERE E1.dst = E2.src 
AND E2.dst = E3.src
AND E3.dst = E1.src
AND E1.dst > E1.src
AND E2.dst > E2.src