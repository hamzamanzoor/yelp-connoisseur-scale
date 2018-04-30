-- Create table for businesses belonging to food categories in US. 
-- (Business - Category Join)

CREATE TABLE us_business (
    auto_ID int NOT NULL auto_increment,
    id varchar(22) NOT NULL,
    name varchar(255) NOT NULL,
    city varchar(255),
    state varchar(255),
    stars float,
    review_count int,
    category varchar(255),
    PRIMARY KEY (auto_ID)
)ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into us_business(id, name, city, state, stars, review_count, category)
 Select distinct a.id, a.name, a.city, a.state, a.stars,
review_count, b.category from category b join business a
on a.id=b.business_id
where b.category in ( 
'Restaurants',
'Food',
'Bars',
'Sandwiches',
'Fast Food',
'American (Traditional)',
'Pizza',
'Coffee & Tea',
'Italian',
'Burgers',
'Breakfast & Brunch',
'Mexican',
'Chinese',
'American (New)',
'Specialty Food',
'Bakeries',
'Cafes',
'Desserts',
'Ice Cream & Frozen Yogurt',
'Japanese',
'Chicken Wings',
'Pubs',
'Seafood',
'Salad',
'Sushi Bars',
'Sports Bars',
'Delis',
'Asian Fusion',
'Mediterranean',
'Barbeque',
'Canadian (New)',
'Steakhouses',
'Indian',
'Thai',
'Juice Bars & Smoothies',
'Diners',
'Middle Eastern',
'French',
'Vegetarian',
'Ethnic Food',
'Buffets',
'Korean',
'Soup',
'Food Trucks',
'Gluten-Free',
'Vegan',
'Tex-Mex',
'Comfort Food',
'Donuts',
'Hot Dogs',
'Bagels',
'Caribbean',
'German',
'Latin American',
'Halal',
'Southern',
'Imported Food',
'Chocolatiers & Shops',
'Tapas/Small Plates',
'Tea Rooms',
'Pakistani',
'Fruits & Veggies',
'British',
'Fish & Chips',
'Noodles',
'Soul Food',
'Hookah Bars',
'Cajun/Creole',
'Modern European',
'Creperies',
'Shaved Ice',
'Swabian',
'Spanish',
'Filipino',
'Cheesesteaks',
'Irish',
'Turkish',
'Patisserie/Cake Shop',
'Lebanese',
'Cheese Shops',
'Taiwanese',
'Custom Cakes',
'Ramen',
'Delicatessen',
'Persian/Iranian',
'Bistros',
'Wineries',
'Tacos',
'Do-It-Yourself Food',
'Food Court',
'African',
'Poutineries',
'International',
'Gelato',
'Kosher',
'Coffee Roasteries',
'Falafel',
'Wraps',
'Afghan',
'Pan Asian',
'Peruvian',
'Waffles',
'Szechuan',
'Kebab',
'Brazilian',
'Scottish',
'Ethiopian')
and a.state in (
'AL',
'AK',
'AZ',
'AR',
'CA',
'CO',
'CT',
'DE',
'FL',
'GA',
'HI',
'ID',
'IL',
'IN',
'IA',
'KS',
'KY',
'LA',
'ME',
'MD',
'MA',
'MI',
'MN',
'MS',
'MO',
'MT',
'NE',
'NV',
'NH',
'NJ',
'NM',
'NY',
'NC',
'ND',
'OH',
'OK',
'OR',
'PA',
'RI',
'SC',
'SD',
'TN',
'TX',
'UT',
'VT',
'VA',
'WA',
'WV',
'WI',
'WY',
'AS',
'DC',
'FM',
'GU',
'MH',
'MP',
'PW',
'PR',
'VI')
;


-- Filter reviews to contain reviews in and after 2015

Create table review_temp
Select * from review where id='----X0BIDP9tA49U3RvdSQ';

ALTER TABLE review_temp
MODIFY COLUMN id varchar(22) PRIMARY KEY ;

ALTER TABLE review_temp ADD INDEX business_id_index (business_id);

Insert ignore into review_temp (id,business_id, user_id, stars, date, text, useful, funny, cool)
Select id,
    business_id, user_id, stars, date,text, useful, funny, cool
FROM review where date>'2015-01-01' 

-- Filter reviews for the eateries in US only

CREATE TABLE `filter_review` (
  `id` varchar(22) NOT NULL,
  `business_id` varchar(22) CHARACTER SET utf8 NOT NULL,
  `user_id` varchar(22) CHARACTER SET utf8 NOT NULL,
  `stars` int(11) DEFAULT NULL,
  `date` datetime DEFAULT NULL,
  `text` mediumtext CHARACTER SET utf8,
  `useful` int(11) DEFAULT NULL,
  `funny` int(11) DEFAULT NULL,
  `cool` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `business_id_index` (`business_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into filter_review (id,
    business_id, user_id, stars, text, useful, funny, cool)
Select id,
    business_id, user_id, stars, text, useful, funny, cool
FROM review_temp where business_id in (select id from temp
);


-- Deploy review_business_join_long Stored procedure (review_business_join_long.sql)
-- Execute review_business_join_long Stored procedure to join businesses with reviews

CREATE TABLE `review_restaurant_final` (
  `auto_ID` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` varchar(22) DEFAULT NULL,
  `business_id` varchar(22) DEFAULT NULL,
  `name` varchar(255) NOT NULL,
  `category` varchar(255) DEFAULT NULL,
  `text` mediumtext,
  `state` varchar(255) DEFAULT NULL,
  `business_rating` float DEFAULT NULL,
  `review_rating` float DEFAULT NULL,
  `useful` int(11) DEFAULT NULL,
  `funny` int(11) DEFAULT NULL,
  `cool` int(11) DEFAULT NULL,
  PRIMARY KEY (`auto_ID`),
  KEY `review_index` (`user_id`,`business_id`),
  KEY `category_index` (`user_id`,`category`),
  KEY `cat_index` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

call review_business_join_long()

-- Calculate frequency of each user on each business category

drop table if exists user_count;
CREATE TABLE `user_count` (
  `user_id` varchar(22) NOT NULL,
  `count` int(11) DEFAULT 0,
  PRIMARY KEY (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into user_count
	Select user_id, count(*) count from 
		review_restaurant_final group by user_id order by 1;

drop table user_fractions;
CREATE TABLE `user_fractions` (
  `user_id` varchar(22) NOT NULL,
  `category` varchar(22) NOT NULL,
  `frequency` float DEFAULT 0.0,
  category_count int DEFAULT 0,
  total_count int DEFAULT 0,
  INDEX user_fractions_index (user_id,category)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


ALTER TABLE user_count ADD INDEX user_count_index (user_id);

Insert into user_fractions
	Select a.user_id, a.category, count(*)/b.count,count(*),b.count from 
		review_restaurant_final a inner join user_count b on a.user_id=b.user_id
			group by a.user_id,a.category;


-- Filtering for 4 categories only

CREATE TABLE auth_business (
    auto_ID int NOT NULL auto_increment,
    id varchar(22) NOT NULL,
    name varchar(255) NOT NULL,
    city varchar(255),
    state varchar(255),
    stars float,
    review_count int,
    category varchar(255),
    PRIMARY KEY (auto_ID)
)ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `review_restaurant_auth` (
  `auto_ID` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` varchar(22) DEFAULT NULL,
  `business_id` varchar(22) DEFAULT NULL,
  `name` varchar(255) NOT NULL,
  `category` varchar(255) DEFAULT NULL,
  `text` mediumtext,
  `state` varchar(255) DEFAULT NULL,
  `business_rating` float DEFAULT NULL,
  `review_rating` float DEFAULT NULL,
  `useful` int(11) DEFAULT NULL,
  `funny` int(11) DEFAULT NULL,
  `cool` int(11) DEFAULT NULL,
  PRIMARY KEY (`auto_ID`),
  KEY `review_index` (`user_id`,`business_id`),
  KEY `category_index` (`user_id`,`category`),
  KEY `cat_index` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into auth_business(id, name, city, state, stars, review_count, category)
select a.id, a.name, a.city, a.state, a.stars, a.review_count, a.category from us_business a 
where a.category in('Italian','Chinese','Indian','Thai')

Insert into review_restaurant_auth
select * from review_restaurant_final where category in('Italian','Chinese','Indian','Thai') 


CREATE TABLE `user_fractions_auth` (
  `user_id` varchar(22) NOT NULL,
  `category` varchar(22) NOT NULL,
  `frequency` float DEFAULT 0.0,
  category_count int DEFAULT 0,
  total_count int DEFAULT 0,
  INDEX user_fractions_auth_index (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into user_fractions_auth
 Select * from user_fractions where user_id in (select distinct user_id from review_restaurant_auth)

-- Filter Friends information
-- Deploy friend_user_join Stored procedure (friend_user_join.sql)
-- Execute friend_user_join Stored procedure to join friends with users to filter the friends 
-- of users in our filtered dataset

drop table if exists friend_filter_auth;
CREATE TABLE `friend_filter_auth` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` varchar(22) NOT NULL,
  `friend_id` varchar(22) NOT NULL,
  PRIMARY KEY (`id`),
  Index friends_index(user_id,friend_id),
  Index friend_index (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

call friend_user_join()


-- Create a table for user experience
-- Experience is number of visits on a restaurants of user divided by the maximum visits

-- distinct users 
CREATE temporary TABLE `uf` (
  `user_id` varchar(22) NOT NULL,
  INDEX uf (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into uf
    Select distinct user_id from `review_restaurant_auth`;


CREATE TABLE `user_bus` (
  `user_id` varchar(22) NOT NULL,
  `business_id` varchar(22) NOT NULL,
  INDEX user_bus (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into user_bus
    Select distinct user_id,business_id from `review_restaurant_final`;


CREATE TABLE `user_exp` (
  `user_id` varchar(22) NOT NULL,
  exp int NOT NULL,
  INDEX user_exp (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

Insert into user_exp
    Select a.user_id, count(*) from user_bus a inner join uf b on a.user_id=b.user_id
        group by a.user_id


SET SESSION group_concat_max_len = 1000000;

SET @SQL = NULL;
SELECT
  GROUP_CONCAT(DISTINCT
    CONCAT(
      'coalesce(sum(case when r.category = ''',
      dt,
      ''' then r.frequency end),0) AS `',
      dt, '`'
    )
  ) INTO @SQL
FROM
(
  SELECT distinct category as dt
  FROM user_fractions_auth s 
) d;


SET @SQL 
  = CONCAT('SELECT r.user_id, ', @SQL, ' 
            from user_fractions_auth r group by r.user_id');
 Select @SQL

PREPARE stmt FROM @SQL;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
Select @SQL;
 
-- The above SQL query gives the following query. Execute it to get 
-- fequency of users on each category in columns

Create table user_freq_category            
SELECT r.user_id, 
    coalesce(sum(case when r.category = 'American (New)' then r.frequency end),0) AS `American (New)`,
    coalesce(sum(case when r.category = 'American (Traditional)' then r.frequency end),0) AS `American (Traditional)`,
    coalesce(sum(case when r.category = 'Bars' then r.frequency end),0) AS `Bars`,
    coalesce(sum(case when r.category = 'British' then r.frequency end),0) AS `British`,
    coalesce(sum(case when r.category = 'Ethnic Food' then r.frequency end),0) AS `Ethnic Food`,
    coalesce(sum(case when r.category = 'Food' then r.frequency end),0) AS `Food`,
    coalesce(sum(case when r.category = 'Indian' then r.frequency end),0) AS `Indian`,
    coalesce(sum(case when r.category = 'Italian' then r.frequency end),0) AS `Italian`,
    coalesce(sum(case when r.category = 'Japanese' then r.frequency end),0) AS `Japanese`,
    coalesce(sum(case when r.category = 'Pubs' then r.frequency end),0) AS `Pubs`,
    coalesce(sum(case when r.category = 'Restaurants' then r.frequency end),0) AS `Restaurants`,
    coalesce(sum(case when r.category = 'Specialty Food' then r.frequency end),0) AS `Specialty Food`,
    coalesce(sum(case when r.category = 'Sports Bars' then r.frequency end),0) AS `Sports Bars`,
    coalesce(sum(case when r.category = 'Steakhouses' then r.frequency end),0) AS `Steakhouses`,
    coalesce(sum(case when r.category = 'Sushi Bars' then r.frequency end),0) AS `Sushi Bars`,
    coalesce(sum(case when r.category = 'Filipino' then r.frequency end),0) AS `Filipino`,
    coalesce(sum(case when r.category = 'Thai' then r.frequency end),0) AS `Thai`,
    coalesce(sum(case when r.category = 'Noodles' then r.frequency end),0) AS `Noodles`,
    coalesce(sum(case when r.category = 'Asian Fusion' then r.frequency end),0) AS `Asian Fusion`,
    coalesce(sum(case when r.category = 'Bakeries' then r.frequency end),0) AS `Bakeries`,
    coalesce(sum(case when r.category = 'Barbeque' then r.frequency end),0) AS `Barbeque`,
    coalesce(sum(case when r.category = 'Breakfast & Brunch' then r.frequency end),0) AS `Breakfast & Brunch`,
    coalesce(sum(case when r.category = 'Burgers' then r.frequency end),0) AS `Burgers`,
    coalesce(sum(case when r.category = 'Cafes' then r.frequency end),0) AS `Cafes`,
    coalesce(sum(case when r.category = 'Cheese Shops' then r.frequency end),0) AS `Cheese Shops`,
    coalesce(sum(case when r.category = 'Cheesesteaks' then r.frequency end),0) AS `Cheesesteaks`,
    coalesce(sum(case when r.category = 'Chicken Wings' then r.frequency end),0) AS `Chicken Wings`,
    coalesce(sum(case when r.category = 'Chinese' then r.frequency end),0) AS `Chinese`,
    coalesce(sum(case when r.category = 'Coffee & Tea' then r.frequency end),0) AS `Coffee & Tea`,
    coalesce(sum(case when r.category = 'Comfort Food' then r.frequency end),0) AS `Comfort Food`,
    coalesce(sum(case when r.category = 'Delis' then r.frequency end),0) AS `Delis`,
    coalesce(sum(case when r.category = 'Fast Food' then r.frequency end),0) AS `Fast Food`,
    coalesce(sum(case when r.category = 'Fish & Chips' then r.frequency end),0) AS `Fish & Chips`,
    coalesce(sum(case when r.category = 'Gluten-Free' then r.frequency end),0) AS `Gluten-Free`,
    coalesce(sum(case when r.category = 'Ice Cream & Frozen Yog' then r.frequency end),0) AS `Ice Cream & Frozen Yog`,
    coalesce(sum(case when r.category = 'Mediterranean' then r.frequency end),0) AS `Mediterranean`,
    coalesce(sum(case when r.category = 'Mexican' then r.frequency end),0) AS `Mexican`,
    coalesce(sum(case when r.category = 'Pizza' then r.frequency end),0) AS `Pizza`,
    coalesce(sum(case when r.category = 'Salad' then r.frequency end),0) AS `Salad`,
    coalesce(sum(case when r.category = 'Sandwiches' then r.frequency end),0) AS `Sandwiches`,
    coalesce(sum(case when r.category = 'Seafood' then r.frequency end),0) AS `Seafood`,
    coalesce(sum(case when r.category = 'Vegan' then r.frequency end),0) AS `Vegan`,
    coalesce(sum(case when r.category = 'Vegetarian' then r.frequency end),0) AS `Vegetarian`,
    coalesce(sum(case when r.category = 'Creperies' then r.frequency end),0) AS `Creperies`,
    coalesce(sum(case when r.category = 'Juice Bars & Smoothies' then r.frequency end),0) AS `Juice Bars & Smoothies`,
    coalesce(sum(case when r.category = 'Buffets' then r.frequency end),0) AS `Buffets`,
    coalesce(sum(case when r.category = 'Southern' then r.frequency end),0) AS `Southern`,
    coalesce(sum(case when r.category = 'Taiwanese' then r.frequency end),0) AS `Taiwanese`,
    coalesce(sum(case when r.category = 'Desserts' then r.frequency end),0) AS `Desserts`,
    coalesce(sum(case when r.category = 'Soul Food' then r.frequency end),0) AS `Soul Food`,
    coalesce(sum(case when r.category = 'Soup' then r.frequency end),0) AS `Soup`,
    coalesce(sum(case when r.category = 'Tapas/Small Plates' then r.frequency end),0) AS `Tapas/Small Plates`,
    coalesce(sum(case when r.category = 'Bagels' then r.frequency end),0) AS `Bagels`,
    coalesce(sum(case when r.category = 'Korean' then r.frequency end),0) AS `Korean`,
    coalesce(sum(case when r.category = 'Hot Dogs' then r.frequency end),0) AS `Hot Dogs`,
    coalesce(sum(case when r.category = 'Pakistani' then r.frequency end),0) AS `Pakistani`,
    coalesce(sum(case when r.category = 'Fruits & Veggies' then r.frequency end),0) AS `Fruits & Veggies`,
    coalesce(sum(case when r.category = 'Tex-Mex' then r.frequency end),0) AS `Tex-Mex`,
    coalesce(sum(case when r.category = 'Gelato' then r.frequency end),0) AS `Gelato`,
    coalesce(sum(case when r.category = 'Middle Eastern' then r.frequency end),0) AS `Middle Eastern`,
    coalesce(sum(case when r.category = 'Cajun/Creole' then r.frequency end),0) AS `Cajun/Creole`,
    coalesce(sum(case when r.category = 'Waffles' then r.frequency end),0) AS `Waffles`,
    coalesce(sum(case when r.category = 'African' then r.frequency end),0) AS `African`,
    coalesce(sum(case when r.category = 'Donuts' then r.frequency end),0) AS `Donuts`,
    coalesce(sum(case when r.category = 'Halal' then r.frequency end),0) AS `Halal`,
    coalesce(sum(case when r.category = 'Chocolatiers & Shops' then r.frequency end),0) AS `Chocolatiers & Shops`,
    coalesce(sum(case when r.category = 'Diners' then r.frequency end),0) AS `Diners`,
    coalesce(sum(case when r.category = 'Food Trucks' then r.frequency end),0) AS `Food Trucks`,
    coalesce(sum(case when r.category = 'French' then r.frequency end),0) AS `French`,
    coalesce(sum(case when r.category = 'German' then r.frequency end),0) AS `German`,
    coalesce(sum(case when r.category = 'Latin American' then r.frequency end),0) AS `Latin American`,
    coalesce(sum(case when r.category = 'Tacos' then r.frequency end),0) AS `Tacos`,
    coalesce(sum(case when r.category = 'Ramen' then r.frequency end),0) AS `Ramen`,
    coalesce(sum(case when r.category = 'Patisserie/Cake Shop' then r.frequency end),0) AS `Patisserie/Cake Shop`,
    coalesce(sum(case when r.category = 'Tea Rooms' then r.frequency end),0) AS `Tea Rooms`,
    coalesce(sum(case when r.category = 'Caribbean' then r.frequency end),0) AS `Caribbean`,
    coalesce(sum(case when r.category = 'Custom Cakes' then r.frequency end),0) AS `Custom Cakes`,
    coalesce(sum(case when r.category = 'Imported Food' then r.frequency end),0) AS `Imported Food`,
    coalesce(sum(case when r.category = 'Modern European' then r.frequency end),0) AS `Modern European`,
    coalesce(sum(case when r.category = 'Wineries' then r.frequency end),0) AS `Wineries`,
    coalesce(sum(case when r.category = 'Irish' then r.frequency end),0) AS `Irish`,
    coalesce(sum(case when r.category = 'Do-It-Yourself Food' then r.frequency end),0) AS `Do-It-Yourself Food`,
    coalesce(sum(case when r.category = 'Szechuan' then r.frequency end),0) AS `Szechuan`,
    coalesce(sum(case when r.category = 'Wraps' then r.frequency end),0) AS `Wraps`,
    coalesce(sum(case when r.category = 'Peruvian' then r.frequency end),0) AS `Peruvian`,
    coalesce(sum(case when r.category = 'Kosher' then r.frequency end),0) AS `Kosher`,
    coalesce(sum(case when r.category = 'Brazilian' then r.frequency end),0) AS `Brazilian`,
    coalesce(sum(case when r.category = 'Lebanese' then r.frequency end),0) AS `Lebanese`,
    coalesce(sum(case when r.category = 'Shaved Ice' then r.frequency end),0) AS `Shaved Ice`,
    coalesce(sum(case when r.category = 'Falafel' then r.frequency end),0) AS `Falafel`,
    coalesce(sum(case when r.category = 'Hookah Bars' then r.frequency end),0) AS `Hookah Bars`,
    coalesce(sum(case when r.category = 'Afghan' then r.frequency end),0) AS `Afghan`,
    coalesce(sum(case when r.category = 'Pan Asian' then r.frequency end),0) AS `Pan Asian`,
    coalesce(sum(case when r.category = 'Turkish' then r.frequency end),0) AS `Turkish`,
    coalesce(sum(case when r.category = 'Food Court' then r.frequency end),0) AS `Food Court`,
    coalesce(sum(case when r.category = 'Spanish' then r.frequency end),0) AS `Spanish`,
    coalesce(sum(case when r.category = 'Ethiopian' then r.frequency end),0) AS `Ethiopian`,
    coalesce(sum(case when r.category = 'Persian/Iranian' then r.frequency end),0) AS `Persian/Iranian`,
    coalesce(sum(case when r.category = 'Coffee Roasteries' then r.frequency end),0) AS `Coffee Roasteries`,
    coalesce(sum(case when r.category = 'Poutineries' then r.frequency end),0) AS `Poutineries`,
    coalesce(sum(case when r.category = 'Bistros' then r.frequency end),0) AS `Bistros`,
    coalesce(sum(case when r.category = 'Kebab' then r.frequency end),0) AS `Kebab`,
    coalesce(sum(case when r.category = 'Delicatessen' then r.frequency end),0) AS `Delicatessen` 
    from user_fractions_auth r group by r.user_id;

-- Create a table of reviews where each review has a maximum experience 
-- of the user who reviewed in on this business. The maximum experience is the total number of reviews 
-- of one user who has maximum reviews among the people who reviewed on that business. 
create table review_exp            
Select d.*,e.max_exp from review_restaurant_auth d left join (
Select c.business_id,max(c.exp) max_exp from
(Select a.user_id,a.business_id,b.exp from review_restaurant_auth a left join user_exp b on a.user_id= b.user_id) c
group by c.business_id) e on d.business_id=e.business_id;