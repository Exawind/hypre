/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifdef USE_NVTX
#include "nvToolsExt.h"
#include "nvToolsExtCudaRt.h"
#include <string>

static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define HYPRE_NVTX_COLORS_BLUE colors[0]
#define HYPRE_NVTX_COLORS_GREEN colors[1]
#define HYPRE_NVTX_COLORS_YELLOW colors[2]
#define HYPRE_NVTX_COLORS_PURPLE colors[3]
#define HYPRE_NVTX_COLORS_TEAL colors[4]
#define HYPRE_NVTX_COLORS_RED colors[5]
#define HYPRE_NVTX_COLORS_WHITE colors[6]

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxDomainRangePushEx(HYPRE_DOMAIN,&eventAttrib);	\
}

#define PUSH_RANGE_PAYLOAD(name,cid,load) {		\
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT64; \
    eventAttrib.payload.llValue = load; \
    eventAttrib.category=1; \
    nvtxDomainRangePushEx(HYPRE_DOMAIN,&eventAttrib); \
}

#define PUSH_RANGE_DOMAIN(name,cid,dId) {				\
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxDomainRangePushEx(getdomain(dId),&eventAttrib);	\
}

#define POP_RANGE nvtxDomainRangePop(HYPRE_DOMAIN);
#define POP_RANGE_DOMAIN(dId) {			\
  nvtxDomainRangePop(getdomain(dId));		\
  }

#define NVTX_MarkEx nvtxMarkEx
#define NVTX_MarkA nvtxMarkA
#define NVTX_MarkW nvtxMarkW

#define NVTX_RangeStartEx nvtxRangeStartEx
#define NVTX_RangeStartA nvtxRangeStartA
#define NVTX_RangeStartW nvtxRangeStartW

#define NVTX_RangeEnd nvtxRangeEnd

#define NVTX_RangePushEx nvtxRangePushEx
#define NVTX_RangePushA nvtxRangePushA
#define NVTX_RangePushW nvtxRangePushW

#define NVTX_RangePop nvtxRangePop

#define NVTX_NameOsThreadA nvtxNameOsThreadA
#define NVTX_NameOsThreadW nvtxNameOsThreadW

#define NVTX_NameCudaStreamA nvtxNameCudaStreamA
#define NVTX_NameCudaStreamW nvtxNameCudaStreamW

#define NVTX_NameCudaStreamA nvtxNameCudaStreamA
#define NVTX_NameCudaStreamW nvtxNameCudaStreamW

#define NVTX_NameCuContextA nvtxNameCuContextA
#define NVTX_NameCuContextW nvtxNameCuContextW

#else

#define PUSH_RANGE(name,cid)
#define POP_RANGE
#define PUSH_RANGE_PAYLOAD(name,cid,load)
#define PUSH_RANGE_DOMAIN(name,cid,domainName)

#define NVTX_MarkEx __noop
#define NVTX_MarkA __noop
#define NVTX_MarkW __noop

#define NVTX_RangeStartEx __noop
#define NVTX_RangeStartA __noop
#define NVTX_RangeStartW __noop

#define NVTX_RangeEnd __noop

#define NVTX_RangePushEx __noop
#define NVTX_RangePushA __noop
#define NVTX_RangePushW __noop

#define NVTX_RangePop __noop

#define NVTX_NameOsThreadA __noop
#define NVTX_NameOsThreadW __noop

#define NVTX_NameCudaStreamA __noop
#define NVTX_NameCudaStreamW __noop

#define NVTX_NameCudaStreamA __noop
#define NVTX_NameCudaStreamW __noop

#define NVTX_NameCuContextA __noop
#define NVTX_NameCuContextW __noop

#endif


// C++ function templates to enable NvToolsExt functions
namespace nvtx {
#ifdef USE_NVTX
class Attributes
{
public:
   Attributes()
   {
      Clear();
   }

   Attributes(const Attributes& other)
   {
      Clear();

      m_Ascii = other.m_Ascii;
      m_Unicode = other.m_Unicode;

      //Copy all the values over
      memcpy(&m_event, other.Out(), other.Out()->size);

      //Copy the correct new pointers over
      m_event.message.ascii = m_Ascii.c_str();
      m_event.message.unicode = m_Unicode.c_str();
   }

   Attributes& Category(uint32_t category)
   {
      m_event.category = category;
      return *this;
   }

   Attributes& Color(uint32_t argb)
   {
      m_event.colorType = NVTX_COLOR_ARGB;
      m_event.color = argb;
      return *this;
   }

   // Attributes& Color(const glm::vec3& rgb)
   // {
   //    m_event.colorType = NVTX_COLOR_ARGB;
   //    m_event.color = glm::packUnorm4x8(glm::vec4(rgb.b, rgb.g, rgb.r, 1.0f));
   //    return *this;
   // }

   Attributes& Payload(uint64_t value)
   {
      m_event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
      m_event.payload.ullValue = value;
      return *this;
   }

   Attributes& Payload(int64_t value)
   {
      m_event.payloadType = NVTX_PAYLOAD_TYPE_INT64;
      m_event.payload.llValue = value;
      return *this;
   }

   Attributes& Payload(uint32_t value)
   {
      m_event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
      m_event.payload.ullValue = value;
      return *this;
   }

   Attributes& Payload(int32_t value)
   {
      m_event.payloadType = NVTX_PAYLOAD_TYPE_INT64;
      m_event.payload.llValue = value;
      return *this;
   }

   Attributes& Payload(double value)
   {
      m_event.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
      m_event.payload.dValue = value;
      return *this;
   }

   Attributes& Message(const std::string& message)
   {
      m_Ascii = message;
      m_event.messageType = NVTX_MESSAGE_TYPE_ASCII;
      m_event.message.ascii = m_Ascii.c_str();
      return *this;
   }

   Attributes& Message(const std::wstring& message)
   {
      m_Unicode = message;
      m_event.messageType = NVTX_MESSAGE_TYPE_UNICODE;
      m_event.message.unicode = m_Unicode.c_str();
      return *this;
   }

   Attributes& Clear()
   {
      memset(&m_event, 0, NVTX_EVENT_ATTRIB_STRUCT_SIZE);
      m_event.version = NVTX_VERSION;
      m_event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      m_Unicode.clear();
      m_Ascii.clear();
      return *this;
   }

   static Attributes Create()
   {
      return Attributes();
   }

   const nvtxEventAttributes_t* Out() const
   {
      return &m_event;
   }

private:
   nvtxEventAttributes_t m_event;

   //Need to hold onto the string values otherwise they will be lost
   std::string m_Ascii;
   std::wstring m_Unicode;
};

class ScopedRange
{
public:
   ScopedRange(const char* message)
   {
      nvtxRangePushA(message);
   }

   ScopedRange(const wchar_t* message)
   {
      nvtxRangePushW(message);
   }

   ScopedRange(const nvtxEventAttributes_t* attributes)
   {
      nvtxRangePushEx(attributes);
   }

   ScopedRange(const Attributes& attributes)
   {
      nvtxRangePushEx(attributes.Out());
   }

   ~ScopedRange()
   {
      nvtxRangePop();
   }
};

inline void Mark(const Attributes& attrib)
{
   nvtxMarkEx(attrib.Out());
}

inline void Mark(const nvtxEventAttributes_t* eventAttrib)
{
   nvtxMarkEx(eventAttrib);
}

inline void Mark(const char* message)
{
   nvtxMarkA(message);
}

inline void Mark(const wchar_t* message)
{
   nvtxMarkW(message);
}

inline nvtxRangeId_t RangeStart(const Attributes& attrib)
{
   return nvtxRangeStartEx(attrib.Out());
}

inline nvtxRangeId_t RangeStart(const nvtxEventAttributes_t* eventAttrib)
{
   return nvtxRangeStartEx(eventAttrib);
}

inline nvtxRangeId_t RangeStart(const char* message)
{
   return nvtxRangeStartA(message);
}

inline nvtxRangeId_t RangeStart(const wchar_t* message)
{
   return nvtxRangeStartW(message);
}

inline void RangeEnd(nvtxRangeId_t id)
{
   nvtxRangeEnd(id);
}

inline int RangePush(const Attributes& attrib)
{
   return nvtxRangePushEx(attrib.Out());
}

inline int RangePush(const nvtxEventAttributes_t* eventAttrib)
{
   return nvtxRangePushEx(eventAttrib);
}

inline int RangePush(const char* message)
{
   return nvtxRangePushA(message);
}

inline int RangePush(const wchar_t* message)
{
   return nvtxRangePushW(message);
}

inline void RangePop()
{
   nvtxRangePop();
}

inline void NameCategory(uint32_t category, const char* name)
{
   nvtxNameCategoryA(category, name);
}

inline void NameCategory(uint32_t category, const wchar_t* name)
{
   nvtxNameCategoryW(category, name);
}

inline void NameOsThread(uint32_t threadId, const char* name)
{
   nvtxNameOsThreadA(threadId, name);
}

inline void NameOsThread(uint32_t threadId, const wchar_t* name)
{
   nvtxNameOsThreadW(threadId, name);
}

inline void NameCurrentThread(const char* name)
{
   NVTX_NameOsThreadA(pthread_self(), name);
}

inline void NameCurrentThread(const wchar_t* name)
{
   NVTX_NameOsThreadW(pthread_self(), name);
}

#else

   typedef uint64_t nvtxRangeId_t;
   typedef struct nvtxEventAttributes_v1 nvtxEventAttributes_t;

   class Attributes
   {
   public:
      Attributes() {}
      Attributes& Category(uint32_t category) { return *this; }
      Attributes& Color(uint32_t argb) { return *this; }
      Attributes& Color(const glm::vec3& rgb) { return *this; }
      Attributes& Payload(uint64_t value) { return *this; }
      Attributes& Payload(int64_t value) { return *this; }
      Attributes& Payload(double value) { return *this; }
      Attributes& Message(const std::string & message) { return *this; }
      Attributes& Message(const std::wstring & message) { return *this; }
      Attributes& Clear() { return *this; }
      static Attributes Create() { return Attributes(); }
      const nvtxEventAttributes_t* Out() { return 0; }
   };

   class ScopedRange
   {
   public:
      ScopedRange(const char* message) { (void)message; }
      ScopedRange(const wchar_t* message) { (void)message; }
      ScopedRange(const nvtxEventAttributes_t* attributes) { (void)attributes; }
      ScopedRange(const Attributes& attributes) { (void)attributes; }
      ~ScopedRange() {}
   };

   inline void Mark(const nvtx::Attributes& attrib) { (void)attrib; }
   inline void Mark(const nvtxEventAttributes_t* eventAttrib) { (void)eventAttrib; }
   inline void Mark(const char* message) { (void)message; }
   inline void Mark(const wchar_t* message) { (void)message; }

   inline nvtxRangeId_t RangeStart(const nvtx::Attributes& attrib) { (void)attrib; return 0; }
   inline nvtxRangeId_t RangeStart(const nvtxEventAttributes_t* eventAttrib) { (void)eventAttrib; return 0; }
   inline nvtxRangeId_t RangeStart(const char* message) { (void)message; return 0; }
   inline nvtxRangeId_t RangeStart(const wchar_t* message) { (void)message; return 0; }

   inline void RangeEnd(nvtxRangeId_t id) { (void)id; }


   inline int RangePush(const nvtx::Attributes& attrib) { (void)attrib; return -1; }
   inline int RangePush(const nvtxEventAttributes_t* eventAttrib) { (void)eventAttrib; return -1; }
   inline int RangePush(const char* message) { (void)message; return -1; }
   inline int RangePush(const wchar_t* message) { (void)message; return -1; }

   inline int RangePop() { return -1; }

   inline void NameCategory(uint32_t category, const char* name) { (void)category; (void)name; }
   inline void NameCategory(uint32_t category, const wchar_t* name) { (void)category; (void)name; }

   inline void NameOsThread(uint32_t threadId, const char* name) { (void)threadId; (void)name; }
   inline void NameOsThread(uint32_t threadId, const wchar_t* name) { (void)threadId; (void)name; }
   inline void NameCurrentThread(const char* name) { (void)name; }
   inline void NameCurrentThread(const wchar_t* name) { (void)name; }

#endif
}

struct adopt_lock_t
{ // indicates adopt lock
};

constexpr adopt_lock_t adopt_lock{};

template<class _Mutex>
class NvLockGuard
{ // specialization for a single mutex
public:
   typedef _Mutex mutex_type;

#ifdef USE_NVTX
   NvLockGuard(_Mutex& _Mtx, const std::string& lockName)
      : _MyName(lockName),
        _MyMutex(_Mtx)
   {
      // construct and lock
      Lock();
   }

   NvLockGuard(_Mutex& _Mtx, const std::string& lockName, const nvtx::Attributes& attributes)
      : _MyName(lockName),
        _MyAttr(attributes),
        _MyMutex(_Mtx)
   {
      // construct and lock
      Lock();
   }
#else
   explicit NvLockGuard(_Mutex& _Mtx)
      : _MyMutex(_Mtx)
   {
   // construct and lock
      _MyMutex.lock();
   }

   NvLockGuard(_Mutex& _Mtx, adopt_lock_t)
      : _MyMutex(_Mtx)
   {
   // construct but don't lock
   }
#endif

   ~NvLockGuard() _NOEXCEPT
   {
#ifdef DO_NVTX_ENABLE
      Unlock();
#endif
      // unlock
      _MyMutex.unlock();
   }

   NvLockGuard(const NvLockGuard&) = delete;
   NvLockGuard& operator=(const NvLockGuard&) = delete;
private:

#ifdef USE_NVTX
   // ReSharper disable once CppMemberFunctionMayBeConst
   void Lock()
   {
      //Mark that we are trying to aquire lock
      nvtxRangeId_t stalledRange = nvtx::RangeStart(nvtx::Attributes(_MyAttr).Message("Lock: " + _MyName + " - Stalled").Color(glm::vec3(1.0, 0.0, 0.0)));

      // construct and lock
      _MyMutex.lock();

      //Mark that we are trying to aquire lock
      nvtx::RangeEnd(stalledRange);

      _MyRangeId = nvtx::RangeStart(nvtx::Attributes(_MyAttr).Message("Lock: " + _MyName + " - Stalled").Color(glm::vec3(0.0, 1.0, 0.0)));
   }

   // ReSharper disable once CppMemberFunctionMayBeConst
   void Unlock()
   {
      nvtx::RangeEnd(_MyRangeId);
   }

   const std::string _MyName;
   nvtx::Attributes _MyAttr;
   nvtxRangeId_t _MyRangeId;
#endif

   _Mutex& _MyMutex;
};

#ifdef USE_NVTX
//Do not use in practice. Use Typed versions
#define HYPRE_NVTX_FULL_LOCK_GUARD(name, mutexType, mut, message, attr) \
   NvLockGuard<mutexType> name(mut, message, attr);

#define HYPRE_NVTX_TYPED_LOCK_GUARD(mutexType, mut, message, attr) \
   HYPRE_NVTX_FULL_LOCK_GUARD(MAKE_UNIQUE_NAME(_nvtx_lock_guard_), mutexType, mut, message, attr)

#define HYPRE_NVTX_RECURSIVE_LOCK_GUARD(mut, message, cat) \
   HYPRE_NVTX_TYPED_LOCK_GUARD(std::recursive_mutex, mut, message, nvtx::Attributes::Create().Category(cat))

#define HYPRE_NVTX_LOCK_GUARD(mut, message, cat) \
   HYPRE_NVTX_TYPED_LOCK_GUARD(std::mutex, mut, message, nvtx::Attributes::Create().Category(cat))

#define HYPRE_NVTX_NAMED_SCOPED_RANGE(name, attr) \
   nvtx::ScopedRange name(attr);

#define HYPRE_NVTX_FULL_SCOPED_RANGE(attr) \
   HYPRE_NVTX_NAMED_SCOPED_RANGE(MAKE_UNIQUE_NAME(_nvtx_auto_scope_), attr)

#define HYPRE_NVTX_COLORED_SCOPED_RANGE(mess, cat, color) \
   HYPRE_NVTX_FULL_SCOPED_RANGE(nvtx::Attributes::Create().Category(cat).Message(mess).Color(color))

#define HYPRE_NVTX_SCOPED_RANGE(mess, cat) \
   HYPRE_NVTX_COLORED_SCOPED_RANGE(mess, cat, glm::vec3(0.0f, 0.0f, 1.0f))

#define HYPRE_NVTX_MARK(message, cat) \
   NVTX_MarkEx(nvtx::Attributes::Create().Message(message).Category(cat));

#else

//Do not use in practice. Use Typed versions
#define HYPRE_NVTX_FULL_LOCK_GUARD(name, mutexType, mut) \
   NvLockGuard<mutexType> name(mut);

//Do not use in practice. Use Typed versions 
#define HYPRE_NVTX_TYPED_LOCK_GUARD(mutexType, mut) \
   HYPRE_NVTX_FULL_LOCK_GUARD(MAKE_UNIQUE_NAME(_nvtx_lock_guard_), mutexType, mut)

#define HYPRE_NVTX_RECURSIVE_LOCK_GUARD(mut, message, cat) \
   HYPRE_NVTX_TYPED_LOCK_GUARD(std::recursive_mutex, mut)

#define HYPRE_NVTX_LOCK_GUARD(mut, message, cat) \
   HYPRE_NVTX_TYPED_LOCK_GUARD(std::mutex, mut)

#define HYPRE_NVTX_NAMED_SCOPED_RANGE(name, attr) __noop

#define HYPRE_NVTX_FULL_SCOPED_RANGE(attr) __noop

#define HYPRE_NVTX_COLORED_SCOPED_RANGE(mess, cat, color) __noop

#define HYPRE_NVTX_SCOPED_RANGE(mess, cat) __noop

#define HYPRE_NVTX_MARK(message, cat) __noop

#endif

//Ensure its a c++ name we are passing not a const char *
#define HYPRE_NVTX_NAME_CURRENT_THREAD(name) \
{ \
   std::string __cpp_name(name); \
   nvtx::NameCurrentThread(__cpp_name.c_str()); \
   SetThreadName(__cpp_name); \
}
