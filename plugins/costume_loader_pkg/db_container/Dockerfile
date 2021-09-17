FROM mariadb:10
ENV MYSQL_ROOT_PASSWORD=test
ENV MYSQL_USER=test
ENV MYSQL_PASSWORD=test
ENV MYSQL_DATABASE=KostuemRepo

COPY mini-muse.sql /docker-entrypoint-initdb.d/
