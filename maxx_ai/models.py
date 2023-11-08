from django.db import models


class Image(models.Model):
    caption=models.CharField(max_length=100)
    image=models.ImageField(upload_to="img/%y")
    def __str__(self):
        return self.caption


class UploadedFile(models.Model):
    # file = models.FileField(upload_to='uploads/photos', default='null')
    file = models.FileField(default='null')
    # video = models.FileField(upload_to='uploads/videos', default='null')
    desc = models.CharField(max_length=200)
    
    def unicode(self):
        return "name{},id{}".format(self.name, self.id)


