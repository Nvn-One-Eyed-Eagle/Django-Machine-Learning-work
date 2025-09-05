from django.shortcuts import render
from .ml.stock_predictor import get_stock_plot

def home(request):
    ticker = request.GET.get("ticker", "AAPL")
    chart, error = get_stock_plot(ticker)

    apple_chart, _ = get_stock_plot("AAPL")  # unpack, ignore error part

    return render(request, "home.html", {
        "chart": chart,
        "ticker": ticker,
        "error": error,
        "apple": apple_chart,   # now this is just the base64 image string
    })
