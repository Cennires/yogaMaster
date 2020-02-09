# Generated by Django 3.0.2 on 2020-02-08 04:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('yogaMaster', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Result',
            fields=[
                ('resultId', models.AutoField(primary_key=True, serialize=False)),
                ('uploadImg', models.ImageField(height_field=360, upload_to='', width_field=200)),
                ('compareImg', models.ImageField(height_field=360, upload_to='', width_field=200)),
                ('content', models.CharField(max_length=255)),
                ('compareTime', models.DateField(auto_now=True)),
            ],
            options={
                'db_table': 'result',
            },
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('usrid', models.AutoField(primary_key=True, serialize=False)),
                ('usrname', models.CharField(max_length=48, null='false')),
                ('password', models.CharField(max_length=48, null='false')),
                ('usrProfile', models.ImageField(height_field=200, upload_to='', width_field=200)),
            ],
            options={
                'db_table': 'user',
            },
        ),
        migrations.CreateModel(
            name='Yoga',
            fields=[
                ('level', models.IntegerField()),
                ('yogaName', models.CharField(max_length=48, primary_key='true', serialize=False)),
                ('video', models.URLField()),
            ],
            options={
                'db_table': 'yoga',
            },
        ),
        migrations.CreateModel(
            name='YogaImage',
            fields=[
                ('imgid', models.AutoField(primary_key=True, serialize=False)),
                ('imgDescription', models.CharField(max_length=255)),
                ('image', models.ImageField(height_field=360, upload_to='', width_field=200)),
                ('yogaName', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='yogaMaster.Yoga')),
            ],
            options={
                'db_table': 'yogaImage',
            },
        ),
        migrations.CreateModel(
            name='StudyRecord',
            fields=[
                ('studyRecordId', models.AutoField(primary_key=True, serialize=False)),
                ('resultid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='yogaMaster.Result')),
                ('usrid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='yogaMaster.User')),
            ],
            options={
                'db_table': 'studyRecord',
            },
        ),
        migrations.AddField(
            model_name='result',
            name='imgid',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='yogaMaster.YogaImage'),
        ),
        migrations.CreateModel(
            name='Favorites',
            fields=[
                ('favoritesId', models.AutoField(primary_key=True, serialize=False)),
                ('imgid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='yogaMaster.YogaImage')),
                ('usrid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='yogaMaster.User')),
            ],
            options={
                'db_table': 'favorites',
            },
        ),
    ]
