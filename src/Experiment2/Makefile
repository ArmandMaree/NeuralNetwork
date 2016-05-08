CC = javac
TARGET = Main
PARAMETERS =

TARGET:
	$(CC) *.java

# Custom commands:
clean:
	rm -f $(TARGET) *.class

run:
	java $(TARGET) $(PARAMETERS)

debug:
	jdb $(TARGET) $(PARAMETERS)

all:
	make
	make run
	make clean

compress:
	make clean
	tar -cvf uXXXXXXXX.tar.gz */*
