/*
 * Copyright (C) 2015 Holmes Team at HUAWEI Noah's Ark Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "Log.h"

Log log_smartDM;

void Log::debug(const char* stringFormat, ...) throw() {
	va_list args;

	printf("[DEBUG] ");
	va_start(args, stringFormat);
	vfprintf(stdout, stringFormat, args);
	va_end(args);
	printf("\n");
}

void Log::debug(const std::string& message) throw() {
	cout << "[DEBUG] " << message << endl;
}

void Log::info(const char* stringFormat, ...) throw() {
	va_list args;

	printf("[INFO] ");
	va_start(args, stringFormat);
	vfprintf(stdout, stringFormat, args);
	va_end(args);
	printf("\n");
}

void Log::info(const std::string& message) throw() {
	cout << "[INFO] " << message << endl;
}

void Log::warn(const char* stringFormat, ...) throw() {
	va_list args;

	printf("[WARN] ");
	va_start(args, stringFormat);
	vfprintf(stdout, stringFormat, args);
	va_end(args);
	printf("\n");
}

void Log::warn(const std::string& message) throw() {
	cout << "[WARN] " << message << endl;
}

void Log::error(const char* stringFormat, ...) throw() {
	va_list args;

	printf("[ERROR] ");
	va_start(args, stringFormat);
	vfprintf(stderr, stringFormat, args);
	va_end(args);
	printf("\n");
}

void Log::error(const std::string& message) throw() {
	cout << "[ERROR] " << message << endl;
}
