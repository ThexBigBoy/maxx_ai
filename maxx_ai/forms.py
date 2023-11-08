from django import forms
from maxx_ai.models import UploadedFile

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ('')
        fields = ['file', 'desc'] 
