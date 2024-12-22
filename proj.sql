SELECT count(Edge.dst) 
FROM Node, Edge
WHERE Node.ID = Edge.src

SELECT Node.ID, Edge.dst
FROM Node, Edge
WHERE Node.ID = Edge.src

SELECT Edge.dst
FROM Node, Edge
WHERE Node.ID = Edge.src