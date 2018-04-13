#include "config.hpp"

using namespace std;
using namespace ocean_ai;

int main() {
	try {
		Config config("config.json");
		int a = 5;
	}
	catch (const std::invalid_argument& ex) {
		cout << "exception: " << ex.what();
	}
	
	return 0;
}