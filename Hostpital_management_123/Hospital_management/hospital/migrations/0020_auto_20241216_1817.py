# Generated by Django 3.0.5 on 2024-12-16 12:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hospital', '0019_auto_20241215_2209'),
    ]

    operations = [
        migrations.AlterField(
            model_name='doctor',
            name='department',
            field=models.CharField(choices=[('Cardiologist', 'Cardiologist'), ('Dermatologists', 'Dermatologists'), ('Emergency Medicine Specialists', 'Emergency Medicine Specialists'), ('Allergists/Immunologists', 'Allergists/Immunologists'), ('Anesthesiologists', 'Anesthesiologists'), ('Colon and Rectal Surgeons', 'Colon and Rectal Surgeons'), ('Neurology', 'Neurology'), ('Orthopaedic', 'Orthopaedic'), ('Pediatrics', 'Pediatrics'), ('Psychiatry', 'Psychiatry'), ('General physician', 'General physician'), ('Gynaecologist', 'Gynaecologist')], default='Cardiologist', max_length=50),
        ),
    ]
