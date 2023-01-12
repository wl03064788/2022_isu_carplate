CREATE DATABASE IF NOT EXISTS `car_violationDB`;
USE `car_violationDB`;

CREATE TABLE IF NOT EXISTS `test` (
  `id` int(20) unsigned NOT NULL AUTO_INCREMENT,
  `time` timestamp NULL DEFAULT current_timestamp(),
  `lattitude` char(30) DEFAULT NULL,
  `longitude` char(30) DEFAULT NULL,
  `plate_number` char(10) DEFAULT NULL,
  `video_address` char(200) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=latin1;

INSERT INTO car_violationdb (latitude , longitude , record_time , plate_number , video_address) VALUES
 ('37° 51.65 S', '145° 7.36 E', '2022-12-14,22-22-53', '6998-0B', 'C:/Users/IoT/Desktop/violation_videos/2022-12-14,22-34-39.avi');