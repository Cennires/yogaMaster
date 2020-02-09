# yogaMaster瑜伽大师（python+django+mysql）

## 基础操作

>1. 使用mysql数据库  
django.db.backends.mysql
>2. 更改时区和语言  
LANGUAGES = [
    ('zh-Hans', _('Chinese')),
]  
LANGUAGE_CODE = 'zh-Hans'  
TIME_ZONE = 'Asia/Shanghai'  
>3. 迁移主要是对model的管理  
Python manage.py makemigrations  
Python manage.py migrate  
>4. 运行项目  
Python manage.py  runserver

## 接口设计

>传输 json 基本格式  
{   
"state": 200, //状态码到时候会给出一个表格，标识各种状态码意义  
"message": "success",   
"data": [  
]   
}


1. Get    http://127.0.0.1:8000/home/getYogaList                
//根据level返回对应的瑜伽列表（初中高代号123）  
request：{"level":"1"}  
Jsonresponse：  
{  
    "state": "200",  
    "message": "获取瑜伽列表成功",  
    "data": "[{\"model\": \"yogaMaster.yoga\", \"pk\": \"ayoga\", \"fields\": {\"level\": 1, \"video\": \"avideourl\"}}, {\"model\": \"yogaMaster.yoga\", \"pk\": \"byoga\", \"fields\": {\"level\": 1, \"video\": \"bvideourl\"}}]"  
}  
2. Get     http://127.0.0.1:8000/home/getYogaDetail          
//根据瑜伽名返回对应的分解动作及视频详情  
3. Post    http://127.0.0.1:8000/home/getResult                      
//用户根据选中的姿势上传图片得到比较结果  
4. Get    http://127.0.0.1:8000/usr/getUsrInfo                         
//获取用户信息  
5. Get    http://127.0.0.1:8000/usr/getStudyRecord                 
//获取用户学习记录  
6. Get    http://127.0.0.1:8000/usr/getFavorites                       
//获取用户收藏

## 数据库设计

表名|属性
-|-
yoga|Level：int，瑜伽等级
-|yogaName：varchar，瑜伽名（pk）
-|video：varchar，瑜伽视频链接
yogaimage|imgid：int，每个动作id（pk）
-|yogaName：varchar，瑜伽名（fk）
-|imgDescription：varchar，每个动作描述
-|image：varchar，每个动作的图片文件路径
user|usrid：int，用户id（pk）
-|usrname：varchar，用户名
-|password：varchar，用户密码
-|usrProfile：varchar，用户头像文件路径
studyrecord|studyRecordId：int，学习记录id（pk）
-|Usrid：int，用户id（fk）
-|resultId：int，比对结果id（fk）
favorites|favoritesId：int，收藏id （pk）
-|Usrid：int，用户id（fk）
-|imgid：int，每个动作id（fk）
result|resultId：int，比对结果id（pk）
-|imgid：int，每个动作id（fk）
-|uploadImg：varchar，用户上传图片文件路径
-|compareImg：varchar，二者比对结果图文件路径
-|content：varchar，二者比对结果描述
-|compareTime：Date，比对时间


## 进度及问题

>关于图片应该是存储到文件夹下，数据库中存储图片路径  
读取的时候先从数据库中读取路径，再到相应路径下读取图片返回  
django生成model时imageFiled在数据库中应该是什么类型，varchar吗  
同一图片不同显示分辨率应需要多个尺寸  
关于数据库的设计还有问题，restful下应该将所有前端需要的资源在一个接口请求中都返回，这样的话要么数据库冗余，要么执行多次查询
