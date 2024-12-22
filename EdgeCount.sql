SELECT count(*)
FROM Node AS Node1, Node AS Node2, Edge
WHERE Edge.src = Node1.ID AND Edge.dst = Node2.ID 
AND Node2.ID > Node1.ID
