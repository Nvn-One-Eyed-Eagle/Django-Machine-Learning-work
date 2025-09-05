from django.shortcuts import render
from .ml.stock_predictor import get_stock_plot

def home(request):
    chart = get_stock_plot()
    return render(request, "home.html", {"chart": chart})