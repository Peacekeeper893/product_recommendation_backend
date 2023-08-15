from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import user , product

from django.http import HttpResponse

from .serializers import userSerializer , productSerializer



# Create your views here.

@api_view(['GET'])
def getRoutes(request):

    routes = [
        {
            'Endpoint': 'api/create-user/',
            'method': 'GET',
            'body': None,
            'description': 'Returns a newly created user id . Should be called if id is not present in the local storage'
        },
        {
            'Endpoint': '/api/add-search-data/',
            'method': 'POST',
            'body': {'userId': "Enter the user id here" , 'q': "Enter the search term here"},
            'description': 'Add Search Interaction to DB for processing'
        },
        {
            'Endpoint': '/api/add-cart-data/',
            'method': 'POST',
            'body': {'userId': "Enter the user id here" , 'q': "Enter the product id here"},
            'description': 'Add Cart Interaction to DB for processing'

        },
        {
            'Endpoint': '/api/recommended/?q=userId',
            'method': 'GET',
            'body': None,
            'description': 'Get recommendations for home page and/or search query page. MAKE SURE TO PASS THE userId as a query param '
        },
        {
            'Endpoint': '/api/search/?q=querystring',
            'method': 'GET',
            'body': None,
            'description': 'Vanilla Search Results based on string matching. MAKE SURE TO PASS THE query typed in search box as a query param '
        },

    ]
    return Response(routes)

@api_view(['GET'])
def createUser(request):
    a = user(name = "test")
    a.save()

    serializer = userSerializer(a)
    return Response(serializer.data)

@api_view(['POST'])
def addSearchData(request ):

    data = request.data

    user_id = int(data['userId'])
    search_term = data['q']
    print("The user id" , user_id)

    curr = user.objects.filter(id = user_id)

    curr_queries = curr[0].search_queries

    user.objects.filter(id = user_id).update(search_queries  = curr_queries + "," + search_term )

    return HttpResponse("Done")


@api_view(['POST'])
def addCartData(request):

    data = request.data

    user_id = int(data['userId'])

    item_id = data['q']

    curr_user = user.objects.filter(id = user_id)
    curr_items = curr_user[0].cart_items

    try:
        curr_item = product.objects.filter(product_uid = item_id)
        prev = curr_item[0].users_interested
        product.objects.filter(product_uid = item_id).update(users_interested = prev+1)
    except:
        print("expected Exception")

    user.objects.filter(id = user_id).update(cart_items  = curr_items + "," + item_id )
    return HttpResponse("Done")


def recommend(request , userId):

    def crazyMLcode():
        pass

    # use the serializer to return the response

def search(request ):

    query = request.GET.get('q')
    searchresults = product.objects.filter(product_title__icontains = query)
    serializer = productSerializer(searchresults , many = True)
    return Response(serializer.data)











