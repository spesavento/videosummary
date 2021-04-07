

/* video table
- short list of the full length videos being analyzed */
CREATE TABLE IF NOT EXISTS video (
  video_id INT AUTO_INCREMENT PRIMARY KEY,
  videoname VARCHAR(30),
  numframes INT UNSIGNED,
  widthpixels INT UNSIGNED,
  heightpixels INT UNSIGNED,
  fps INT UNSIGNED,
  PRIMARY KEY (video_id)
) ENGINE = InnoDB;

/* boundingbox table
- storing objects detected (people, objects, text) and bounding box */
CREATE TABLE IF NOT EXISTS boundingbox (
    boundingbox_id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT UNSIGNED,
    frame_number INT UNSIGNED,
    object_detected VARCHAR(30),
    confidence VARCHAR(30),
    bb_width INT UNSIGNED,
    bb_height INT UNSIGNED,
    bb_left INT UNSIGNED,
    bb_top INT UNSIGNED,
    PRIMARY KEY (boundingbox_id) )
ENGINE = InnoDB;
