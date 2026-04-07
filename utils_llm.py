import os
import requests
import threading
import time
import html
import re
import datetime
import email.utils
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from html.parser import HTMLParser

# redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

WEATHER_CACHE_TTL_S = int(os.environ.get('WEATHER_CACHE_TTL_S', 600))
WEATHER_FETCH_RETRY_GUARD_S = float(os.environ.get('WEATHER_FETCH_RETRY_GUARD_S', 20))
EVENTS_CACHE_TTL_S = int(os.environ.get('EVENTS_CACHE_TTL_S', 60 * 60 * 3))  # 3 hours
EVENTS_FETCH_RETRY_GUARD_S = float(os.environ.get('EVENTS_FETCH_RETRY_GUARD_S', 20))
EVENTS_FEED_URL = os.environ.get('EVENTS_FEED_URL', 'https://today.wisc.edu/events/feed/463')
EVENTS_TIMEZONE = os.environ.get('EVENTS_TIMEZONE', 'America/Chicago')
EVENTS_MAX_PER_DAY = int(os.environ.get('EVENTS_MAX_PER_DAY', 3))
EVENTS_LOCATION_FILTER = os.environ.get('EVENTS_LOCATION_FILTER', 'Morgridge Hall')
DEFAULT_BROWSER_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/122.0.0.0 Safari/537.36'
)
EVENTS_USER_AGENT = os.environ.get('EVENTS_USER_AGENT', DEFAULT_BROWSER_USER_AGENT)
WEATHER_USER_AGENT = os.environ.get('WEATHER_USER_AGENT', DEFAULT_BROWSER_USER_AGENT)

_weather_lock = threading.Lock()
_weather_cache = {
    'value': '',
    'expires_at': 0.0,
    'last_attempt_at': 0.0,
}
_events_lock = threading.Lock()
_events_cache = {
    'value': '',
    'expires_at': 0.0,
    'last_attempt_at': 0.0,
}


def _strip_html(value):
    if not value:
        return ''
    text = re.sub(r'<[^>]+>', ' ', str(value))
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _parse_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt is not None:
            return dt
    except Exception:
        pass

    try:
        cleaned = text.replace('Z', '+00:00')
        return datetime.datetime.fromisoformat(cleaned)
    except Exception:
        pass

    for fmt in [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%Y %I:%M %p',
        '%m/%d/%Y',
    ]:
        try:
            return datetime.datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _xml_local_name(tag):
    if not isinstance(tag, str):
        return ''
    if '}' in tag:
        return tag.split('}', 1)[1].lower()
    return tag.lower()


def _find_child_text(item, candidate_names):
    names = set(x.lower() for x in candidate_names)
    for child in list(item):
        local_name = _xml_local_name(child.tag)
        if local_name in names:
            text = (child.text or '').strip()
            if text:
                return text
    return ''


def _extract_link(item):
    for child in list(item):
        local_name = _xml_local_name(child.tag)
        if local_name != 'link':
            continue
        href = child.attrib.get('href', '').strip()
        if href:
            return href
        text = (child.text or '').strip()
        if text:
            return text
    return ''


def _extract_event_date(item):
    candidates = []
    for child in list(item):
        local_name = _xml_local_name(child.tag)
        if local_name in [
            'start', 'startdate', 'start_date', 'dtstart', 'eventstart', 'eventdate',
            'published', 'updated', 'pubdate', 'date'
        ]:
            text = (child.text or '').strip()
            if text:
                candidates.append(text)

    for raw in candidates:
        dt = _parse_datetime(raw)
        if dt is None:
            continue
        if dt.tzinfo is None:
            return dt
        try:
            return dt.astimezone(ZoneInfo(EVENTS_TIMEZONE))
        except Exception:
            return dt
    return None


def _parse_events_feed(feed_xml):
    if not isinstance(feed_xml, str) or not feed_xml.strip():
        return []
    try:
        root = ET.fromstring(feed_xml)
    except Exception:
        return _parse_events_html_listing(feed_xml)

    items = root.findall('.//item')
    if not items:
        items = root.findall('.//{http://www.w3.org/2005/Atom}entry')

    events = []
    for item in items:
        title = _find_child_text(item, ['title'])
        if not title:
            continue
        event_dt = _extract_event_date(item)
        link = _extract_link(item)
        description = _find_child_text(item, ['description', 'summary'])
        events.append(
            {
                'title': _strip_html(title),
                'date': event_dt,
                'link': link,
                'description': _strip_html(description),
            }
        )
    if events:
        return events
    return _parse_events_html_listing(feed_xml)


def _parse_heading_date(text):
    if not text:
        return None
    cleaned = ' '.join(str(text).split())
    if not cleaned:
        return None

    try:
        now = datetime.datetime.now(ZoneInfo(EVENTS_TIMEZONE))
    except Exception:
        now = datetime.datetime.now()

    lowered = cleaned.lower()
    relative_prefixes = {
        'today': 0,
        'tomorrow': 1,
        'yesterday': -1,
    }
    for token, day_delta in relative_prefixes.items():
        if lowered == token:
            picked_date = (now + datetime.timedelta(days=day_delta)).date()
            return datetime.datetime.combine(picked_date, datetime.time.min)
        marker = f'{token},'
        if lowered.startswith(marker):
            remainder = cleaned[len(marker):].strip()
            if not remainder:
                picked_date = (now + datetime.timedelta(days=day_delta)).date()
                return datetime.datetime.combine(picked_date, datetime.time.min)
            cleaned = remainder
            lowered = cleaned.lower()
            break

    for fmt in ['%A, %B %d, %Y', '%a, %b %d, %Y']:
        try:
            dt = datetime.datetime.strptime(cleaned, fmt)
            return dt
        except Exception:
            continue
    for fmt in ['%B %d, %Y', '%b %d, %Y']:
        try:
            dt = datetime.datetime.strptime(cleaned, fmt)
            return dt
        except Exception:
            continue
    for fmt in ['%A, %B %d', '%a, %b %d']:
        try:
            parsed = datetime.datetime.strptime(cleaned, fmt)
            now_year = datetime.datetime.now().year
            return parsed.replace(year=now_year)
        except Exception:
            continue
    for fmt in ['%B %d', '%b %d']:
        try:
            parsed = datetime.datetime.strptime(cleaned, fmt)
            now_year = datetime.datetime.now().year
            return parsed.replace(year=now_year)
        except Exception:
            continue
    return None


def _attrs_class_list(attrs):
    for name, value in attrs:
        if (name or '').lower() == 'class' and value:
            return str(value).lower().split()
    return []


def _class_has(attrs, token):
    token = str(token or '').strip().lower()
    if not token:
        return False
    return token in _attrs_class_list(attrs)


def _clean_event_time(value):
    text = _strip_html(value)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*-\s*', '-', text)
    return text


def _clean_event_location(value):
    text = _strip_html(value)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\bAlso offered online\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip(' ,;|-')
    return text


def _is_morgridge_event(event):
    filter_text = str(EVENTS_LOCATION_FILTER or '').strip().lower()
    if not filter_text:
        return True
    location = str(event.get('location', '') or '').lower()
    description = str(event.get('description', '') or '').lower()
    title = str(event.get('title', '') or '').lower()
    haystack = ' '.join([location, description, title])
    return filter_text in haystack


class _EventsHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_date = None
        self.events = []
        self.in_event = False
        self.event_depth = 0
        self.current_event = None
        self.capture_mode = None
        self.capture_tag = None
        self.capture_text = []

    def handle_starttag(self, tag, attrs):
        tag = (tag or '').lower()

        if self.in_event:
            self.event_depth += 1

        if tag == 'h2' and _class_has(attrs, 'day-row-header'):
            self.capture_mode = 'date'
            self.capture_tag = 'h2'
            self.capture_text = []
            return

        if tag == 'li' and _class_has(attrs, 'event-row'):
            self.in_event = True
            self.event_depth = 1
            self.current_event = {
                'title': '',
                'date': self.current_date,
                'link': '',
                'description': '',
                'time': '',
                'location': '',
            }
            return

        if not self.in_event:
            return

        if tag == 'h3' and _class_has(attrs, 'event-title'):
            self.capture_mode = 'title'
            self.capture_tag = 'h3'
            self.capture_text = []
            return
        if tag == 'p' and _class_has(attrs, 'event-time'):
            self.capture_mode = 'time'
            self.capture_tag = 'p'
            self.capture_text = []
            return
        if tag == 'p' and _class_has(attrs, 'event-location'):
            self.capture_mode = 'location'
            self.capture_tag = 'p'
            self.capture_text = []
            return

        if tag == 'a' and self.capture_mode == 'title' and self.current_event is not None:
            href = ''
            for name, value in attrs:
                if (name or '').lower() == 'href':
                    href = str(value or '').strip()
                    break
            if href and not self.current_event.get('link'):
                self.current_event['link'] = href

    def handle_startendtag(self, tag, attrs):
        tag = (tag or '').lower()
        if self.capture_mode in ['time', 'location'] and tag == 'br':
            self.capture_text.append(' ')

    def handle_data(self, data):
        if self.capture_mode and data:
            self.capture_text.append(data)

    def handle_endtag(self, tag):
        tag = (tag or '').lower()

        if self.capture_mode and tag == self.capture_tag:
            text = _strip_html(' '.join(self.capture_text))
            if self.capture_mode == 'date':
                parsed_date = _parse_heading_date(text)
                if parsed_date is not None:
                    self.current_date = parsed_date
            elif self.current_event is not None:
                if self.capture_mode == 'title':
                    self.current_event['title'] = text
                elif self.capture_mode == 'time':
                    self.current_event['time'] = _clean_event_time(text)
                elif self.capture_mode == 'location':
                    self.current_event['location'] = _clean_event_location(text)
            self.capture_mode = None
            self.capture_tag = None
            self.capture_text = []

        if not self.in_event:
            return

        self.event_depth -= 1
        if self.event_depth <= 0:
            if self.current_event is not None:
                title = str(self.current_event.get('title', '')).strip()
                event_date = self.current_event.get('date')
                if title and isinstance(event_date, datetime.datetime):
                    self.events.append(self.current_event)
            self.current_event = None
            self.in_event = False
            self.event_depth = 0


def _parse_events_html_listing(html_text):
    if not isinstance(html_text, str) or not html_text.strip():
        return []
    parser = _EventsHtmlParser()
    try:
        parser.feed(html_text)
    except Exception:
        return []
    return parser.events


def _build_events_summary(events):
    if not isinstance(events, list):
        return ''
    try:
        now = datetime.datetime.now(ZoneInfo(EVENTS_TIMEZONE))
    except Exception:
        now = datetime.datetime.now()
    today = now.date()
    tomorrow = today + datetime.timedelta(days=1)

    today_items = []
    tomorrow_items = []
    for ev in events:
        if not _is_morgridge_event(ev):
            continue
        ev_date = ev.get('date')
        if isinstance(ev_date, datetime.datetime):
            ev_day = ev_date.date()
        else:
            continue
        if ev_day == today:
            today_items.append(ev)
        elif ev_day == tomorrow:
            tomorrow_items.append(ev)

    def _format(day_items):
        if not day_items:
            return 'none found'
        formatted = []
        for ev in day_items[:max(1, EVENTS_MAX_PER_DAY)]:
            title = str(ev.get('title', '')).strip()
            event_time = str(ev.get('time', '')).strip()
            location = str(ev.get('location', '')).strip()
            if title:
                details = []
                if event_time:
                    details.append(event_time)
                if location:
                    details.append(location)
                if details:
                    formatted.append(f"{title} ({', '.join(details)})")
                else:
                    formatted.append(title)
        if not formatted:
            return 'none found'
        return '; '.join(formatted)

    return f"Today: {_format(today_items)} | Tomorrow: {_format(tomorrow_items)}"


def _fetch_weather():
    # res = redis_client.get('cache:weather_info')
    # if res is not None:
    #     print('[check_weather] USING CACHE!')
    #     return res

    print('[check_weather] RUNNING!')
    madison_url = 'https://api.weather.gov/gridpoints/MKX/38,64/forecast'
    headers = {
        "User-Agent": WEATHER_USER_AGENT,
        "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
    }
    weather_str = ''
    try:
        forecast_response = requests.get(madison_url, headers=headers, timeout=6)
        forecast_data = forecast_response.json()
        weather_str = ''
        weather_now = forecast_data['properties']['periods'][0]
        weather_next = forecast_data['properties']['periods'][1]
        weather_str += f"{weather_now['name']}: {weather_now['detailedForecast']}\n"
        weather_str += f"{weather_next['name']}: {weather_next['detailedForecast']}"
        return weather_str
    except Exception as e:
        print(f'[check_weather] ERROR: {str(e)}')
    return ''


def check_weather(allow_refresh=True, only_if_missing=False):
    now = time.time()
    with _weather_lock:
        cached_value = _weather_cache.get('value', '') or ''
        expires_at = float(_weather_cache.get('expires_at', 0.0))
        has_fresh_cache = bool(cached_value) and expires_at > now
        if has_fresh_cache:
            return cached_value

        if not allow_refresh:
            return cached_value

        if only_if_missing and cached_value:
            return cached_value

        last_attempt_at = float(_weather_cache.get('last_attempt_at', 0.0))
        if (now - last_attempt_at) < WEATHER_FETCH_RETRY_GUARD_S:
            return cached_value
        _weather_cache['last_attempt_at'] = now

    fresh_value = _fetch_weather()

    with _weather_lock:
        if fresh_value:
            _weather_cache['value'] = fresh_value
            _weather_cache['expires_at'] = time.time() + max(60, WEATHER_CACHE_TTL_S)
            return fresh_value
        return _weather_cache.get('value', '') or ''


def warm_weather_cache_if_needed():
    return check_weather(allow_refresh=True, only_if_missing=False)


def _fetch_events_summary():
    print('[check_events] RUNNING!')
    headers = {
        "User-Agent": EVENTS_USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Referer": "https://today.wisc.edu/",
    }
    try:
        resp = requests.get(EVENTS_FEED_URL, headers=headers, timeout=7)
        feed_text = resp.text
        events = _parse_events_feed(feed_text)
        summary = _build_events_summary(events)
        return summary
    except Exception as e:
        print(f'[check_events] ERROR: {str(e)}')
    return ''


def check_events_summary(allow_refresh=True, only_if_missing=False):
    now = time.time()
    with _events_lock:
        cached_value = _events_cache.get('value', '') or ''
        expires_at = float(_events_cache.get('expires_at', 0.0))
        has_fresh_cache = bool(cached_value) and expires_at > now
        if has_fresh_cache:
            return cached_value

        if not allow_refresh:
            return cached_value

        if only_if_missing and cached_value:
            return cached_value

        last_attempt_at = float(_events_cache.get('last_attempt_at', 0.0))
        if (now - last_attempt_at) < EVENTS_FETCH_RETRY_GUARD_S:
            return cached_value
        _events_cache['last_attempt_at'] = now

    fresh_value = _fetch_events_summary()

    with _events_lock:
        if fresh_value:
            _events_cache['value'] = fresh_value
            _events_cache['expires_at'] = time.time() + max(60, EVENTS_CACHE_TTL_S)
            return fresh_value
        return _events_cache.get('value', '') or ''


def warm_events_cache_if_needed():
    return check_events_summary(allow_refresh=True, only_if_missing=False)
