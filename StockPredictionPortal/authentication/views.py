from django.shortcuts import render, redirect
from .forms import UserSignUpForm

def signup(request):
    if request.method == 'POST':
        form = UserSignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserSignUpForm()
    return render(request, "registration/signup.html", {"form": form})
