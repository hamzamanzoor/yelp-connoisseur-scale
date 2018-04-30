DELIMITER $$
 DROP PROCEDURE IF EXISTS review_business_join_long$$
 CREATE PROCEDURE review_business_join_long()
 BEGIN
 DECLARE x  INT;
 DECLARE y  INT;
 DECLARE i  INT;
 DECLARE j  INT;
 
 SET x = (select count(*) from us_business);
 SET y = (select count(*) from filter_review); 
 SET i = 0;
 SET j = 0;

 WHILE i  < x DO
	
    drop table if exists temp;
    CREATE TABLE temp (
		auto_ID int NOT NULL auto_increment,
		id varchar(22) NOT NULL,
		name varchar(255) NOT NULL,
		city varchar(255),
		state varchar(255),
		stars float,
		review_count int,
		category varchar(255),
		PRIMARY KEY (auto_ID),
        INDEX temp_index (id)
	)ENGINE=InnoDB DEFAULT CHARSET=latin1;
	Insert ignore into temp
		Select * from us_business order by id limit i,10000;
	SET j = 0;    
    WHILE j < y DO  
		drop table if exists temp1;
        CREATE TABLE `temp1` (
		  `id` varchar(22) NOT NULL,
		  `business_id` varchar(22) CHARACTER SET utf8 NOT NULL,
		  `user_id` varchar(22) CHARACTER SET utf8 NOT NULL,
		  `stars` int(11) DEFAULT NULL,
		  `text` mediumtext CHARACTER SET utf8,
		  `useful` int(11) DEFAULT NULL,
		  `funny` int(11) DEFAULT NULL,
		  `cool` int(11) DEFAULT NULL,
		  KEY `temp1_index` (`business_id`)
		) ENGINE=InnoDB DEFAULT CHARSET=latin1;
        
		Insert ignore into temp1
			Select * from filter_review limit j,100000;
		Insert into review_restaurant_final(user_id, business_id, name, category, text, business_rating, review_rating,useful,funny,cool)
			Select a.user_id,a.business_id, b.name, b.category, a.text, b.stars, a.stars, a.useful,a.funny, a.cool
				from temp1 a inner join temp b
					on a.business_id=b.id;

		SET  j = j + 1 + 100000; 
		Select concat('Current: ',i,' ', j);
        DO SLEEP(2);
	END WHILE;
	SET  i = i + 1 + 10000;  
 END WHILE;
 
 END$$
DELIMITER ;