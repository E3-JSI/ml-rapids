#ifndef API_H_
#define API_H_

#if defined(_MSC_VER)
    #ifdef STREAMDM_EXPORTS
        #define STREAMDM_API __declspec(dllexport)
    #else
        #define STREAMDM_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__)
    #ifdef STREAMDM_EXPORTS
        #define STREAMDM_API __attribute__((visibility("default")))
    #else
        #define STREAMDM_API
    #endif
#else
    #define STREAMDM_API
#endif

#endif /* API_H_ */
