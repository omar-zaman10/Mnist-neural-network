CXX = g++
CXXFLAGS = -std=c++17 -I include
TARGET = main
OBJ_DIR = obj
SRC_DIR = src


SRCS = $(wildcard $(SRC_DIR)/*.cc)
OBJS = $(addprefix $(OBJ_DIR)/, $(notdir $(SRCS:.cc=.o)))

all: $(TARGET)

# Link the executable from the object files in OBJ_DIR
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean