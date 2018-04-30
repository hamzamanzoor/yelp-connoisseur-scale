DELIMITER $$
 DROP PROCEDURE IF EXISTS friend_user_join$$
 CREATE PROCEDURE friend_user_join()
 BEGIN
 DECLARE x  INT;
 DECLARE i  INT;
 
 SET x = (select count(*) from friend);
 SET i = 0;

create temporary table tt
select distinct user_id from user_fractions_auth;

 WHILE i  < x DO
	
    drop table if exists temp;
    CREATE TABLE temp (
		`id` int(11) NOT NULL AUTO_INCREMENT,
	    `user_id` varchar(22) NOT NULL,
	    `friend_id` varchar(22) NOT NULL,
	    PRIMARY KEY (`id`),
	    Index friends_index(user_id,friend_id)
	) ENGINE=InnoDB DEFAULT CHARSET=utf8;
    
	Insert ignore into temp (user_id,friend_id)
		Select user_id,friend_id from friend limit i,500000;
        
    Insert ignore into friend_filter_auth (user_id,friend_id)
		select user_id,friend_id from temp where user_id in (select user_id from tt);
	SET  i = i + 1 + 500000;  
    Select concat('Current: ',i);
	DO SLEEP(2); 
 END WHILE;
 
 drop temporary table if exists tt;
 
 END$$
DELIMITER ;

