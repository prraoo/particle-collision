#include <exception>
#include <iostream>
#include <iomanip>
#include <filesystem>

#include <utils/argparse.h>

#include "HDRDemo.h"

using namespace std::literals;


namespace
{
	std::ostream& print_usage(std::ostream& out)
	{
		return out << R"""(usage: hdr_pipeline [{options}] {<input-file>}
	options:
	  --device <i>           use CUDA device <i>, default: 0
	  --exposure <v>         set exposure value to <v>, default: 0.0
	  --brightpass <v>       set brightpass threshold to <v>, default: 0.9
	  --test-runs <N>        average timings over <N> test runs, default: 1
)""";
	}
}

int main(int argc, char* argv[])
{
	try
	{
		HDRDemo demo;
		std::filesystem::path envmap;
		int cuda_device = 0;
		float exposure_value = 0.0f;
		float brightpass_threshold = 0.9f;
		int test_runs = 1;

		for (const char* const* a = argv + 1; *a; ++a)
		{
			if (!argparse::parseIntArgument(cuda_device, a, "--device"sv))
			if (!argparse::parseFloatArgument(exposure_value, a, "--exposure"sv))
			if (!argparse::parseFloatArgument(brightpass_threshold, a, "--brightpass"sv))
			if (!argparse::parseIntArgument(test_runs, a, "--test-runs"))
			{
				std::filesystem::path input_file = *a;

				auto ext = input_file.extension();

				if (ext == ".hdr"sv || ext == ".pfm"sv)
					envmap = std::move(input_file);
				else
					demo.add_model(input_file);
			}
		}

		if (envmap.empty())
			throw argparse::usage_error("expected input file");

		if (test_runs < 0)
			throw argparse::usage_error("number of test runs cannot be negative");

		demo.run(envmap.filename().replace_extension(".png"), envmap, cuda_device, exposure_value, brightpass_threshold, test_runs);
	}
	catch (const argparse::usage_error& e)
	{
		std::cerr << "ERROR: " << e.what() << '\n' << print_usage;
		return -127;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << '\n';
		return -1;
	}
	catch (...)
	{
		std::cerr << "ERROR: unknown exception\n";
		return -128;
	}

	return 0;
}
