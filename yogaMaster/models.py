from django.db import models


# Create your models here.
class User(models.Model):
    class Meta:
        db_table = 'user'

    usrid = models.AutoField(primary_key='true')
    usrname = models.CharField(max_length=48, null='false')
    password = models.CharField(max_length=48, null='false')
    usrProfile = models.ImageField(upload_to='avater')


class Yoga(models.Model):
    class Meta:
        db_table = 'yoga'

    level = models.IntegerField()
    imgid = models.AutoField(primary_key='true')
    yogaName = models.CharField(max_length=48)
    video = models.URLField()


# 用户上传图片记录
# class YogaImage(models.Model):
#     class Meta:
#         db_table = 'yogaImage'
#     # 添加用户记录
#     userId = models.ForeignKey('user', on_delete=models.CASCADE())
#     # 记录上传对应标准图片的id，方便进行对应模型评估
#     imgid = models.ForeignKey('yoga', on_delete=models.CASCADE())
#     yogaNameByUser = models.CharField(max_length=48)
#     imgDescription = models.CharField(max_length=255)
#     image = models.ImageField(upload_to='yoga')
    # upload_to='photos' 这句表示上传的文件会存放在$MEDIA_ROOT/photos/ 下面
    # MEDIA_ROOT = D:\研一\挑战杯\yogaMaster\yogaMaster/images/


class Result(models.Model):
    class Meta:
        db_table = 'result'

    resultId = models.AutoField(primary_key='true')
    imgid = models.ForeignKey('yoga', on_delete=models.CASCADE)
    # 上传照片
    uploadImg = models.ImageField(upload_to='upload')
    # 返回节点标识照片
    compareImg = models.ImageField(upload_to='result')
    # 返回json文件
    jsonFile = models.URLField()
    # 生成npy文件
    npyFile = models.URLField()
    # 反馈信息
    content = models.CharField(max_length=255)
    # 完成时间
    compareTime = models.DateField(auto_now='true')


class StudyRecord(models.Model):
    class Meta:
        db_table = 'studyRecord'

    studyRecordId = models.AutoField(primary_key='true')
    usrid = models.ForeignKey('user', on_delete=models.CASCADE)
    resultid = models.ForeignKey('result', on_delete=models.CASCADE)


class Favorites(models.Model):
    class Meta:
        db_table = 'favorites'

    favoritesId = models.AutoField(primary_key='true')
    usrid = models.ForeignKey('user', on_delete=models.CASCADE)
    imgid = models.ForeignKey('yoga', on_delete=models.CASCADE)
