Invoke-WebRequest -Uri https://zenodo.org/record/1161203/files/data.tar.gz -OutFile data.tar.gz
tar -zxvf data.tar.gz
Remove-Item data.tar.gz