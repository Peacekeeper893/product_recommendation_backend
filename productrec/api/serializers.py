from rest_framework.serializers import ModelSerializer
from .models import user , product

class userSerializer(ModelSerializer):
    class Meta:
        model = user
        fields = '__all__'

class productSerializer(ModelSerializer):
    class Meta:
        model = product
        fields = '__all__'

