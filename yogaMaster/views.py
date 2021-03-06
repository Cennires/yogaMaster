import os

from _pytest import logging

# Create your views here.
# -*- coding: utf-8 -*-
from django.core import serializers
from django.http import HttpRequest, HttpResponseBadRequest, JsonResponse, HttpResponse
import simplejson
from django.utils import timezone

from yoga import settings
from .models import User, Yoga, YogaImage, Result, StudyRecord, Favorites
from .parse import parse_sequence,load_ps
from .evaluate import evaluate_pose


# Create your views here.

# Get     /home/getYogaList                //根据level返回对应的瑜伽列表（初中高代号123）
def getYogaList(request: HttpRequest):
    print(request.body)
    try:
        payload = simplejson.loads(request.body)
        level = payload['level']
        yogalist = Yoga.objects.filter(level=level)
        print(yogalist)
        return JsonResponse({
            'state': '200',
            'message': '获取瑜伽列表成功',
            'data': serializers.serialize('json', yogalist, ensure_ascii=False)
        })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


# Get     /home/getYogaDetail           //根据每个瑜伽动作文件名返回对应的图片
def getYogaDetail(request: HttpRequest):
    print(request.body)
    try:
        urls=''
        payload = simplejson.loads(request.body)
        name = payload['yogaName']
        yogaImglist = YogaImage.objects.filter(yogaName=name)
        for var in yogaImglist:
            print(str(var.image))
            imagepath = os.path.join(settings.WEB_HOST_MEDIA_URL, "yogaMaster/images/{}".format(str(var.image)))
            urls += imagepath+'[/--sp--/]'
        return JsonResponse({
            'state': '200',
            'message': '获取瑜伽图片列表成功',
            'data': urls[:len(urls) - len('[/--sp--/]')]
        })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


# Get    /usr/getUsrAvater                        //获取用户头像
def getUsrAvater(request: HttpRequest):
    print(request.body)
    try:
        payload = simplejson.loads(request.body)
        usrid = payload['usrid']
        user = User.objects.get(usrid=usrid)
        print(settings.BASE_DIR)
        print(user.usrProfile)
        imagepath = os.path.join(settings.BASE_DIR, "yogaMaster/images/{}".format(str(user.usrProfile)))
        image_data = picture(imagepath)
        return HttpResponse(image_data, content_type="image/png")
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


# Get    /usr/getUsrInfo                        //获取用户信息
def getUsrInfo(request: HttpRequest):
    print(request.body)
    try:
        payload = simplejson.loads(request.body)
        usrid = payload['usrid']
        user = User.objects.filter(usrid=usrid)
        print(user)
        return JsonResponse({
            'state': '200',
            'message': '获取用户信息成功',
            'data': serializers.serialize('json', user, ensure_ascii=False)
        })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


# post    /usr/register    //用户注册
def register(request: HttpRequest):
    try:
        user = User()
        user.usrname = request.POST.get('usrname')
        user.password = request.POST.get('password')
        user.usrProfile = request.FILES.get('usrProfile')
        user.save()
        print(user)
        return JsonResponse({
            'state': '200',
            'message': '注册成功',
        })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


# Post    /home/getResult                    //用户根据选中的姿势上传图片得到比较结果
def getResult(request: HttpRequest):
    try:
        result = Result()
        imgid = request.POST.get('imgid')
        result.imgid = Yoga.objects.get(imgid=imgid)
        # original = Yoga.objects.get(imgid=imgid)
        result.uploadImg = request.FILES.get('uploadimg')
        # result.compareImg = compareYoga(result.uploadImg.url, original.url)
        result.compareImg, result.jsonFile = openpose(result.uploadImg)
        result.npyFile, result.content = compareYoga(result.jsonFile, imgid)
        result.compareTime = timezone.now()
        result.save()
        imagepath = os.path.join(settings.BASE_DIR, "yogaMaster/images/{}".format(str(result.compareImg)))
        image_data = picture(imagepath)
        studyrecord = StudyRecord()
        studyrecord.resultid = result.resultId
        studyrecord.usrid = request.user.usrid
        return HttpResponse(image_data, content_type="image/png")
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


def compareYoga(uploadimg: str, imgid):
    print('compareYoga....')
    # str：json路径；imgid:以数字来表示，代表desk_keypoints等不同标识
    path = str
    parse_sequence(path, imgid)
    path2 = os.path.join(os.getcwd(), 'json_npy')
    npyFilePath = os.path.join(path2,imgid,'.npy')
    pose_seq = load_ps(npyFilePath)
    # 评测动作
    # (correct, feedback) = evaluate_pose(pose_seq, 'upside_angle')
    (correct, feedback) = evaluate_pose(pose_seq, imgid)
    advice = ''
    if correct:
        advice += 'Exercise performed correctly!'
    else:
        advice += 'Exercise could be improved:'
    content = feedback+'\n'+advice
    print(content)
    return npyFilePath, content


def openpose(uploading: str):
    print("openpose.............")
    # 填充openpose算法
    url = ''
    compareImg = "result/{}".format('a.jpg')
    return compareImg, url


# Get    /usr/getStudyRecord               //获取用户学习记录
def getStudyRecord(request: HttpRequest):
    print(request.body)
    try:
        urls=''
        payload = simplejson.loads(request.body)
        usrid = payload['usrid']
        result = StudyRecord.objects.filter(usrid=usrid)
        for sr in result:
            res = Result.objects.get(resultId=sr.resultid.resultId)
            imagepath = os.path.join(settings.WEB_HOST_MEDIA_URL, "yogaMaster/images/{}".format(str(res.compareImg)))
            urls += imagepath + '[/--sp--/]'
        return JsonResponse({
            'state': '200',
            'message': '获取学习记录成功',
            'data': urls[:len(urls) - len('[/--sp--/]')]
        })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


# Get    /usr/getFavorites                     //获取用户收藏
def getFavorites(request: HttpRequest):
    print(request.body)
    try:
        urls=''
        payload = simplejson.loads(request.body)
        usrid = payload['usrid']
        result = Favorites.objects.filter(usrid=usrid)
        for fa in result:
            res = YogaImage.objects.get(imgid=fa.imgid.imgid)
            print(res)
            imagepath = os.path.join(settings.WEB_HOST_MEDIA_URL, "yogaMaster/images/{}".format(str(res.image)))
            urls += imagepath + '[/--sp--/]'
        return JsonResponse({
            'state': '200',
            'message': '获取收藏列表成功',
            'data': urls[:len(urls) - len('[/--sp--/]')]
        })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()

#下均为未完成接口
# Get     /usr/getAllUsr                //获取全部用户信息列表
def getAllUsr(request: HttpRequest):
    print(request.body)
    try:
        return JsonResponse({
            'state': '200',
            'message': '获取全部用户信息成功',
           })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()

# Get     /usr/login                //后台管理员登录，可写死用户名密码admin/admin
def login(request: HttpRequest):
    print(request.body)
    try:
        return JsonResponse({
            'state': '200',
            'message': '登录成功',
           })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()

# Get     /home/getAllYoga                //返回全部的瑜伽列表
def getAllYoga(request: HttpRequest):
    print(request.body)
    try:
        return JsonResponse({
            'state': '200',
            'message': '获取瑜伽列表成功',
           })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()

# Post     /home/addYoga                //在后端管理页面上传新的瑜伽信息
def addYoga(request: HttpRequest):
    print(request.body)
    try:
        return JsonResponse({
            'state': '200',
            'message': '瑜伽信息上传成功',
           })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()

# Get     /home/getAllResult                //获取全部的结果比对图片
def addYoga(request: HttpRequest):
    print(request.body)
    try:
        return JsonResponse({
            'state': '200',
            'message': '获取全部结果比对信息成功',
           })
    except Exception as e:
        # logging.info(e)
        print(e)
        return HttpResponseBadRequest()


def picture(imagepath):
    print(imagepath)
    with open(imagepath, 'rb') as f:
        image_data = f.read()
    return image_data
