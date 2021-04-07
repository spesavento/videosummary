
/* this is a dump of SQL commands used on MySQL Workbench */

/* create "video" table
- short list of the full length videos being analyzed */
CREATE TABLE IF NOT EXISTS videometadata.video (
  video_id INT AUTO_INCREMENT PRIMARY KEY,
  videoname VARCHAR(30),
  numframes INT UNSIGNED,
  widthpixels INT UNSIGNED,
  heightpixels INT UNSIGNED,
  fps INT UNSIGNED
) ENGINE = InnoDB;

/* create "boundingbox" table
- storing objects detected (people, objects, text) and bounding box
- this information is derived from AWS rekognition
*/
CREATE TABLE IF NOT EXISTS videometadata.boundingbox (
    boundingbox_id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT UNSIGNED,
    frame_number INT UNSIGNED,
    object_detected VARCHAR(30),
    confidence FLOAT,
    bb_width FLOAT,
    bb_height FLOAT,
    bb_left FLOAT,
    bb_top FLOAT
) ENGINE = InnoDB;

/* misc SQL that works using SQL Workbench */
USE videometadata;
SELECT * FROM boundingbox;
ALTER TABLE boundingbox MODIFY COLUMN confidence FLOAT;
DELETE from boundingbox WHERE boundingbox_id BETWEEN 42 AND 59;
SELECT * FROM boundingbox WHERE object_detected = 'Bicycle' and video_id = 1 ORDER BY frame_number ASC;

