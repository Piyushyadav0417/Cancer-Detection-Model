from django import forms

class PredictionForm(forms.Form):
    image = forms.ImageField(label='Upload Mouth Image')
    model_choice = forms.ChoiceField(
        label='Choose Model:',
        choices=[
            ('one.h5', 'Model One (Large Dataset)'),
            ('two.h5', 'Model Two (Lightweight)')
        ]
    )
