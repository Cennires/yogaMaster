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

### 小程序接口设计

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
  //根据每个瑜伽动作文件名返回对应的图片  
  request: {"yogaName":"ayoga"}   
  Jsonresponse：  
  {  
  ​    "state": "200",  
  ​    "message": "获取瑜伽图片列表成功",  
  ​    "data": "http://127.0.0.1:8000/yogaMaster/images/yoga/2.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/3.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/4.jpg"  
  }

3. Get    http://127.0.0.1:8000/usr/getUsrInfo  
  //获取用户信息  
  request:   {"usrid":"1"}  
  Jsonresponse：  
  {  
    "state": "200",  
    "message": "获取用户信息成功",  
    "data": "[{\"model\": \"yogaMaster.user\", \"pk\": 1, \"fields\": {\"usrname\": \"yy\", \"password\": \"abc\", \"usrProfile\": \"yogaMaster/images/avater/2.jpg\"}}]"  
  }

4. Get    http://127.0.0.1:8000/usr/getUsrAvater  
  //获取用户头像  
  request: {"userid":"1"}  
  HttpResponse(image_data, content_type="image/png")  

5. Post    http://127.0.0.1:8000/usr/register  
  //注册  
  var form = new FormData();  
  form.append("usrProfile", fileInput.files[0], "/C:/Users/yang/Desktop/2.jpg");  
  form.append("usrname", "yy");  
  form.append("password", "abc");  
  Jsonresponse：  
  {  
    "state": "200",  
    "message": "注册成功"  
  }

6. Post    http://127.0.0.1:8000/home/getResult  
  //用户根据选中的姿势上传图片得到比较结果图片  
  requset :  
  var form = new FormData();  
  form.append("imgid", "1");  
  form.append("uploadimg", fileInput.files[0], "/C:/Users/yang/Desktop/3.png");  
  HttpResponse(image_data, content_type="image/png")  

7. Get    http://127.0.0.1:8000/usr/getStudyRecord  
  //获取用户学习记录  
  request:
  {"usrid":"1"}  
  Jsonresponse：  
   {  
  ​    "state": "200",  
  ​    "message": "获取学习记录成功",  
  ​    "data": "http://127.0.0.1:8000/yogaMaster/images/result/a.jpg"  
  }

8. Get    http://127.0.0.1:8000/usr/getFavorites  
  //获取用户收藏  
  request:   {"usrid":"1"}  
  Jsonresponse：  
  {  
  ​    "state": "200",  
  ​    "message": "获取收藏列表成功",  
  ​    "data": "http://127.0.0.1:8000/yogaMaster/images/yoga/1.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/2.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/3.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/4.jpg"  
  }

### 后台管理网站接口设计

1. Get    http://127.0.0.1:8000/usr/getUsrInfo  
  //获取用户信息  
  request:    {"usrid":"1"}  
  Jsonresponse：  
  {  
    "state": "200",  
    "message": "获取用户信息成功",  
    "data": "[{\"model\": \"yogaMaster.user\", \"pk\": 1, \"fields\": {\"usrname\": \"yy\", \"password\": \"abc\", \"usrProfile\": \"yogaMaster/images/avater/2.jpg\"}}]"  
  }

2. Get    http://127.0.0.1:8000/usr/getStudyRecord  
  //获取用户学习记录  
  request:
  {"usrid":"1"}  
  Jsonresponse：  
   {  
  ​    "state": "200",  
  ​    "message": "获取学习记录成功",  
  ​    "data": "http://127.0.0.1:8000/yogaMaster/images/result/a.jpg"  
  }

3. Get    http://127.0.0.1:8000/usr/getFavorites  
  //获取用户收藏  
  request:   {"usrid":"1"}  
  Jsonresponse：  
  {  
  ​    "state": "200",  
  ​    "message": "获取收藏列表成功",  
  ​    "data": "http://127.0.0.1:8000/yogaMaster/images/yoga/1.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/2.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/3.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/4.jpg"  
  }

4. (新增) Get    http://127.0.0.1:8000/usr/getAllUsr  
  //获取全部用户信息列表  
  Jsonresponse：  
  {  
    "state": "200",  
    "message": "获取全部用户信息成功",  
    "data": "[{\"model\": \"yogaMaster.user\", \"pk\": 1, \"fields\": {\"usrname\": \"yy\", \"password\": \"abc\", \"usrProfile\": \"yogaMaster/images/avater/2.jpg\"}，{\"usrname\": \"yy1\", \"password\": \"abc\", \"usrProfile\": \"yogaMaster/images/avater/1.jpg\"}}]"  
  }

5. (新增)  Get    http://127.0.0.1:8000/usr/login  
  //后台管理员登录，可写死用户名密码admin/admin  
  request:   
  {"usrname":"admin","password":"admin"}   
  Jsonresponse：  
   {  
  ​    "state": "200",  
  ​    "message": "登录成功"
  }

6. (新增) Get    http://127.0.0.1:8000/home/getAllYoga                
  //返回全部的瑜伽列表  
  Jsonresponse：  
  {  
    "state": "200",  
    "message": "获取瑜伽列表成功",  
    "data": "[{\"model\": \"yogaMaster.yoga\", \"pk\": \"ayoga\", \"fields\": {\"level\": 1, \"video\": \"avideourl\"}}, {\"model\": \"yogaMaster.yoga\", \"pk\": \"byoga\", \"fields\": {\"level\": 1, \"video\": \"bvideourl\"}}]"  
  }  

7. Get     http://127.0.0.1:8000/home/getYogaDetail          
  //根据每个瑜伽动作文件名返回对应的图片  
  request: {"yogaName":"ayoga"}   
  Jsonresponse：  
  {  
  ​    "state": "200",  
  ​    "message": "获取瑜伽图片列表成功",  
  ​    "data": "http://127.0.0.1:8000/yogaMaster/images/yoga/2.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/3.jpg[/--sp--/]http://127.0.0.1:8000/yogaMaster/images/yoga/4.jpg"  
  }

8. (新增)  Post    http://127.0.0.1:8000/home/addYoga  
  //在后端管理页面上传新的瑜伽信息 
  requset :  
  var form = new FormData();  
  form.append("imgid", "1");  
  form.append("uploadimg", fileInput.files[0], "/C:/Users/yang/Desktop/3.png");  
  Jsonresponse：  
  {  
  ​    "state": "200",  
  ​    "message": "瑜伽图信息上传成功" 
  }  

9. (新增)  Get    http://127.0.0.1:8000/home/getAllResult  
  //获取全部的结果比对图片  
  Jsonresponse：  
   {  
  ​    "state": "200",  
  ​    "message": "获取全部结果比对信息成功",  
  ​    "data": "[{\"model\": \"yogaMaster.Result\", \"pk\": \"ayoga\", \"fields\": {\"level\": 1, \"video\": \"avideourl\"}}, {\"model\": \"yogaMaster.Result\", \"pk\": \"byoga\", \"fields\": {\"level\": 1, \"video\": \"bvideourl\"}}]"  
  }


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

>对图片资源的请求单独列出来以httpresponse的形式返回，文字以json形式返回  
关于图片应该是存储到文件夹下，数据库中存储图片路径  
读取的时候先从数据库中读取路径，再到相应路径下读取图片返回  
django生成model时imageFiled在数据库中就是imageField，读取时imageFiled.url  
同一图片不同显示分辨率应需要多个尺寸  
关于数据库的设计还有问题，restful下应该将所有前端需要的资源在一个接口请求中都返回，这样的话要么数据库冗余，要么执行多次查询
