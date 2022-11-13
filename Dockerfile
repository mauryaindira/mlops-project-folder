FROM python:3.8.1
RUN mkdir exp
COPY ./*.py /exp/
COPY ./requirement.txt /exp/requirement.txt
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
#CMD ["export", "FLASK_APP=app.py ;"," flask run"]
CMD ["python3","./plot_graphs.py"]
