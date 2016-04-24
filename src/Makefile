CC = g++
CFLAGS =
LFLAGS = -static
TARGET = CharacterRecognition
OBJECTS = Main.o NeuralNetwork.o Neuron.o

# Linking all the object code:
main: $(OBJECTS)
	$(CC) $(LFLAGS) $(OBJECTS) -o $(TARGET)

# Dependencies:
Main.o: Main.cpp
NeuralNetwork.o: NeuralNetwork.cpp
Neuron.o: Neuron.cpp


Compilation rule:
%.o: %.cpp
	$(CC) $(CFLAGS) -c $?

# Custom commands:
clean:
	rm -f $(TARGET) $(OBJECTS) *.gch *~ # deleting executable, .o’s and backups

run:
	./$(TARGET) # executing the program

valgrind:
	make clean
	make
	valgrind --tool=memcheck ./$(TARGET)
	make clean

all:
	make clean
	make
	make run
	make clean
