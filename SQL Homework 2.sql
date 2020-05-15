.read data.sql

-- Q2
CREATE TABLE obedience as
  -- REPLACE THIS LINE
  SELECT seven, gerald FROM students;


-- Q3
CREATE TABLE blue_dog as
  -- REPLACE THIS LINE
  SELECT color, pet FROM students WHERE color = 'blue' AND pet = 'dog';


-- Q4
CREATE TABLE smallest_int as
  -- REPLACE THIS LINE
  SELECT time, smallest FROM students WHERE smallest > 3 ORDER BY smallest LIMIT 20;


-- Q5
CREATE TABLE sevens as
  -- REPLACE THIS LINE
  SELECT s.number - c.'7' FROM students as s, checkboxes as c WHERE s.number = 7 AND c.'7' = 'True' AND s.time = c.time;


-- Q6
CREATE TABLE matchmaker as
  -- REPLACE THIS LINE
  SELECT first.pet, first.song, first.color, second.color FROM students as first, students as second WHERE first.pet = second.pet AND first.song = second.song AND first.time < second.time;


-- Q7
CREATE TABLE smallest_int_count as
  -- REPLACE THIS LINE
  SELECT smallest, COUNT(smallest) FROM students WHERE smallest > 0 GROUP BY smallest;
