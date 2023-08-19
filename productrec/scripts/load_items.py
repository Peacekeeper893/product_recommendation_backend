
from api.models import product
import csv


def run():
    with open('api/fashion.csv' , encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Advance past the header

        product.objects.all().delete()


        for row in reader:

            # print(type(row[4]))

            pr = product(product_uid=row[0],product_title=row[1] ,product_img_url = row[2] ,product_description = row[3] , product_price = row[4] )
            pr.save()
            
