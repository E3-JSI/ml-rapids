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

#ifndef Log_H_
#define Log_H_

#include <iostream>
#include <string>
#include <stdarg.h>
#include "../API.h"

#ifdef _MSC_VER
#define LOG_DEBUG ::log_smartDM.debug
#define LOG_INFO ::log_smartDM.info
#define LOG_WARN ::log_smartDM.warn
#define LOG_ERROR ::log_smartDM.error
#else
#define LOG_DEBUG(x, args...) log_smartDM.debug(x, ##args)
#define LOG_INFO(x, args...) log_smartDM.info(x, ##args)
#define LOG_WARN(x, args...) log_smartDM.warn(x, ##args)
#define LOG_ERROR(x, args...) log_smartDM.error(x, ##args)
#endif

using namespace std;

class STREAMDM_API Log {
public:

	/**
	 * Log a message with debug priority.
	 * @param stringFormat Format specifier for the string to write
	 * in the log file.
	 * @param ... The arguments for stringFormat
	 **/
	void debug(const char* stringFormat, ...) throw();

	/**
	 * Log a message with debug priority.
	 * @param message string to write in the log file
	 **/
	void debug(const std::string& message) throw();

	/**
	 * Log a message with info priority.
	 * @param stringFormat Format specifier for the string to write
	 * in the log file.
	 * @param ... The arguments for stringFormat
	 **/
	void info(const char* stringFormat, ...) throw();

	/**
	 * Log a message with info priority.
	 * @param message string to write in the log file
	 **/
	void info(const std::string& message) throw();

	/**
	 * Log a message with warn priority.
	 * @param stringFormat Format specifier for the string to write
	 * in the log file.
	 * @param ... The arguments for stringFormat
	 **/
	void warn(const char* stringFormat, ...) throw();

	/**
	 * Log a message with warn priority.
	 * @param message string to write in the log file
	 **/
	void warn(const std::string& message) throw();

	/**
	 * Log a message with error priority.
	 * @param stringFormat Format specifier for the string to write
	 * in the log file.
	 * @param ... The arguments for stringFormat
	 **/
	void error(const char* stringFormat, ...) throw();

	/**
	 * Log a message with error priority.
	 * @param message string to write in the log file
	 **/
	void error(const std::string& message) throw();

};

extern Log log_smartDM;

#endif /* Log_H_ */
