from django.db import models

# Create your models here.

class user( models.Model ):
    id = models.AutoField(primary_key=True)
    name = models.CharField(null = True , max_length=50 )
    age = models.IntegerField(null=True)
    search_queries = models.TextField(default="" , max_length=10000)
    cart_items = models.TextField(default="" , max_length=10000)

    def __str__(self):
        return self.name[0:50]
    

class product(models.Model):
    product_uid = models.TextField(primary_key=True , max_length=100)
    product_title = models.TextField(max_length=100)
    product_img_url = models.TextField(max_length=800)
    users_interested = models.IntegerField(default=0)
    def __str__(self):
        return self.product_title[0:50]

