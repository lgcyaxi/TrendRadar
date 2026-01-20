# coding=utf-8
"""
本地存储后端 - SQLite + TXT/HTML

使用 SQLite 作为主存储，支持可选的 TXT 快照和 HTML 报告
"""

import sqlite3
import shutil
import pytz
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from trendradar.storage.base import StorageBackend, NewsItem, NewsData, RSSItem, RSSData
from trendradar.storage.sqlite_mixin import SQLiteStorageMixin
from trendradar.utils.time import (
    get_configured_time,
    format_date_folder,
    format_time_filename,
)


class LocalStorageBackend(SQLiteStorageMixin, StorageBackend):
    """
    本地存储后端

    使用 SQLite 数据库存储新闻数据，支持：
    - 按日期组织的 SQLite 数据库文件
    - 可选的 TXT 快照（用于调试）
    - HTML 报告生成
    """

    def __init__(
        self,
        data_dir: str = "output",
        enable_txt: bool = True,
        enable_html: bool = True,
        timezone: str = "Asia/Shanghai",
    ):
        """
        初始化本地存储后端

        Args:
            data_dir: 数据目录路径
            enable_txt: 是否启用 TXT 快照
            enable_html: 是否启用 HTML 报告
            timezone: 时区配置（默认 Asia/Shanghai）
        """
        self.data_dir = Path(data_dir)
        self.enable_txt = enable_txt
        self.enable_html = enable_html
        self.timezone = timezone
        self._db_connections: Dict[str, sqlite3.Connection] = {}

    @property
    def backend_name(self) -> str:
        return "local"

    @property
    def supports_txt(self) -> bool:
        return self.enable_txt

    # ========================================
    # SQLiteStorageMixin 抽象方法实现
    # ========================================

    def _get_configured_time(self) -> datetime:
        """获取配置时区的当前时间"""
        return get_configured_time(self.timezone)

    def _format_date_folder(self, date: Optional[str] = None) -> str:
        """格式化日期文件夹名 (ISO 格式: YYYY-MM-DD)"""
        return format_date_folder(date, self.timezone)

    def _format_time_filename(self) -> str:
        """格式化时间文件名 (格式: HH-MM)"""
        return format_time_filename(self.timezone)

    def _get_db_path(self, date: Optional[str] = None, db_type: str = "news") -> Path:
        """
        获取 SQLite 数据库路径

        新结构（扁平）：output/{type}/{date}.db
        - output/news/2025-12-28.db
        - output/rss/2025-12-28.db

        Args:
            date: 日期字符串
            db_type: 数据库类型 ("news" 或 "rss")

        Returns:
            数据库文件路径
        """
        date_str = self._format_date_folder(date)
        db_dir = self.data_dir / db_type
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / f"{date_str}.db"

    def _get_connection(self, date: Optional[str] = None, db_type: str = "news") -> sqlite3.Connection:
        """
        获取数据库连接（带缓存）

        Args:
            date: 日期字符串
            db_type: 数据库类型 ("news" 或 "rss")

        Returns:
            数据库连接
        """
        db_path = str(self._get_db_path(date, db_type))

        if db_path not in self._db_connections:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self._init_tables(conn, db_type)
            self._db_connections[db_path] = conn

        return self._db_connections[db_path]

    # ========================================
    # StorageBackend 接口实现（委托给 mixin）
    # ========================================

    def save_news_data(self, data: NewsData) -> bool:
        """保存新闻数据到 SQLite"""
        db_path = self._get_db_path(data.date)
        if not db_path.exists():
            # 确保目录存在
            db_path.parent.mkdir(parents=True, exist_ok=True)

        success, new_count, updated_count, title_changed_count, off_list_count = \
            self._save_news_data_impl(data, "[本地存储]")

        if success:
            # 输出详细的存储统计日志
            log_parts = [f"[本地存储] 处理完成：新增 {new_count} 条"]
            if updated_count > 0:
                log_parts.append(f"更新 {updated_count} 条")
            if title_changed_count > 0:
                log_parts.append(f"标题变更 {title_changed_count} 条")
            if off_list_count > 0:
                log_parts.append(f"脱榜 {off_list_count} 条")
            print("，".join(log_parts))

        return success

    def get_today_all_data(self, date: Optional[str] = None) -> Optional[NewsData]:
        """获取指定日期的所有新闻数据（合并后）"""
        db_path = self._get_db_path(date)
        if not db_path.exists():
            return None
        return self._get_today_all_data_impl(date)

    def get_latest_crawl_data(self, date: Optional[str] = None) -> Optional[NewsData]:
        """获取最新一次抓取的数据"""
        db_path = self._get_db_path(date)
        if not db_path.exists():
            return None
        return self._get_latest_crawl_data_impl(date)

    def detect_new_titles(self, current_data: NewsData) -> Dict[str, Dict]:
        """检测新增的标题"""
        return self._detect_new_titles_impl(current_data)

    def is_first_crawl_today(self, date: Optional[str] = None) -> bool:
        """检查是否是当天第一次抓取"""
        db_path = self._get_db_path(date)
        if not db_path.exists():
            return True
        return self._is_first_crawl_today_impl(date)

    def get_crawl_times(self, date: Optional[str] = None) -> List[str]:
        """获取指定日期的所有抓取时间列表"""
        db_path = self._get_db_path(date)
        if not db_path.exists():
            return []
        return self._get_crawl_times_impl(date)

    def has_pushed_today(self, date: Optional[str] = None) -> bool:
        """检查指定日期是否已推送过"""
        return self._has_pushed_today_impl(date)

    def record_push(self, report_type: str, date: Optional[str] = None) -> bool:
        """记录推送"""
        success = self._record_push_impl(report_type, date)
        if success:
            now_str = self._get_configured_time().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[本地存储] 推送记录已保存: {report_type} at {now_str}")
        return success

    # ========================================
    # RSS 数据存储方法
    # ========================================

    def save_rss_data(self, data: RSSData) -> bool:
        """保存 RSS 数据到 SQLite"""
        success, new_count, updated_count = self._save_rss_data_impl(data, "[本地存储]")

        if success:
            # 输出统计日志
            log_parts = [f"[本地存储] RSS 处理完成：新增 {new_count} 条"]
            if updated_count > 0:
                log_parts.append(f"更新 {updated_count} 条")
            print("，".join(log_parts))

        return success

    def get_rss_data(self, date: Optional[str] = None) -> Optional[RSSData]:
        """获取指定日期的所有 RSS 数据"""
        return self._get_rss_data_impl(date)

    def detect_new_rss_items(self, current_data: RSSData) -> Dict[str, List[RSSItem]]:
        """检测新增的 RSS 条目"""
        return self._detect_new_rss_items_impl(current_data)

    def get_latest_rss_data(self, date: Optional[str] = None) -> Optional[RSSData]:
        """获取最新一次抓取的 RSS 数据"""
        db_path = self._get_db_path(date, db_type="rss")
        if not db_path.exists():
            return None
        return self._get_latest_rss_data_impl(date)

    # ========================================
    # 本地特有功能：TXT/HTML 快照
    # ========================================

    def save_txt_snapshot(self, data: NewsData) -> Optional[str]:
        """
        保存 TXT 快照

        新结构：output/txt/{date}/{time}.txt

        Args:
            data: 新闻数据

        Returns:
            保存的文件路径
        """
        if not self.enable_txt:
            return None

        try:
            date_folder = self._format_date_folder(data.date)
            txt_dir = self.data_dir / "txt" / date_folder
            txt_dir.mkdir(parents=True, exist_ok=True)

            file_path = txt_dir / f"{data.crawl_time}.txt"

            with open(file_path, "w", encoding="utf-8") as f:
                for source_id, news_list in data.items.items():
                    source_name = data.id_to_name.get(source_id, source_id)

                    # 写入来源标题
                    if source_name and source_name != source_id:
                        f.write(f"{source_id} | {source_name}\n")
                    else:
                        f.write(f"{source_id}\n")

                    # 按排名排序
                    sorted_news = sorted(news_list, key=lambda x: x.rank)

                    for item in sorted_news:
                        line = f"{item.rank}. {item.title}"
                        if item.url:
                            line += f" [URL:{item.url}]"
                        if item.mobile_url:
                            line += f" [MOBILE:{item.mobile_url}]"
                        f.write(line + "\n")

                    f.write("\n")

                # 写入失败的来源
                if data.failed_ids:
                    f.write("==== 以下ID请求失败 ====\n")
                    for failed_id in data.failed_ids:
                        f.write(f"{failed_id}\n")

            print(f"[本地存储] TXT 快照已保存: {file_path}")
            return str(file_path)

        except Exception as e:
            print(f"[本地存储] 保存 TXT 快照失败: {e}")
            return None

    def save_html_report(self, html_content: str, filename: str, is_summary: bool = False) -> Optional[str]:
        """
        保存 HTML 报告

        新结构：output/html/{date}/{filename}

        Args:
            html_content: HTML 内容
            filename: 文件名
            is_summary: 是否为汇总报告

        Returns:
            保存的文件路径
        """
        if not self.enable_html:
            return None

        try:
            date_folder = self._format_date_folder()
            html_dir = self.data_dir / "html" / date_folder
            html_dir.mkdir(parents=True, exist_ok=True)

            file_path = html_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"[本地存储] HTML 报告已保存: {file_path}")
            return str(file_path)

        except Exception as e:
            print(f"[本地存储] 保存 HTML 报告失败: {e}")
            return None

    # ========================================
    # 本地特有功能：资源清理
    # ========================================

    def cleanup(self) -> None:
        """清理资源（关闭数据库连接）"""
        for db_path, conn in self._db_connections.items():
            try:
                conn.close()
                print(f"[本地存储] 关闭数据库连接: {db_path}")
            except Exception as e:
                print(f"[本地存储] 关闭连接失败 {db_path}: {e}")

        self._db_connections.clear()

    def cleanup_old_data(self, retention_days: int) -> int:
        """
        清理过期数据

        新结构清理逻辑：
        - output/news/{date}.db  -> 删除过期的 .db 文件
        - output/rss/{date}.db   -> 删除过期的 .db 文件
        - output/txt/{date}/     -> 删除过期的日期目录
        - output/html/{date}/    -> 删除过期的日期目录

        Args:
            retention_days: 保留天数（0 表示不清理）

        Returns:
            删除的文件/目录数量
        """
        if retention_days <= 0:
            return 0

        deleted_count = 0
        cutoff_date = self._get_configured_time() - timedelta(days=retention_days)

        def parse_date_from_name(name: str) -> Optional[datetime]:
            """从文件名或目录名解析日期 (ISO 格式: YYYY-MM-DD)"""
            # 移除 .db 后缀
            name = name.replace('.db', '')
            try:
                date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', name)
                if date_match:
                    return datetime(
                        int(date_match.group(1)),
                        int(date_match.group(2)),
                        int(date_match.group(3)),
                        tzinfo=pytz.timezone(self.timezone)
                    )
            except Exception:
                pass
            return None

        try:
            if not self.data_dir.exists():
                return 0

            # 清理数据库文件 (news/, rss/)
            for db_type in ["news", "rss"]:
                db_dir = self.data_dir / db_type
                if not db_dir.exists():
                    continue

                for db_file in db_dir.glob("*.db"):
                    file_date = parse_date_from_name(db_file.name)
                    if file_date and file_date < cutoff_date:
                        # 先关闭数据库连接
                        db_path = str(db_file)
                        if db_path in self._db_connections:
                            try:
                                self._db_connections[db_path].close()
                                del self._db_connections[db_path]
                            except Exception:
                                pass

                        # 删除文件
                        try:
                            db_file.unlink()
                            deleted_count += 1
                            print(f"[本地存储] 清理过期数据: {db_type}/{db_file.name}")
                        except Exception as e:
                            print(f"[本地存储] 删除文件失败 {db_file}: {e}")

            # 清理快照目录 (txt/, html/)
            for snapshot_type in ["txt", "html"]:
                snapshot_dir = self.data_dir / snapshot_type
                if not snapshot_dir.exists():
                    continue

                for date_folder in snapshot_dir.iterdir():
                    if not date_folder.is_dir() or date_folder.name.startswith('.'):
                        continue

                    folder_date = parse_date_from_name(date_folder.name)
                    if folder_date and folder_date < cutoff_date:
                        try:
                            shutil.rmtree(date_folder)
                            deleted_count += 1
                            print(f"[本地存储] 清理过期数据: {snapshot_type}/{date_folder.name}")
                        except Exception as e:
                            print(f"[本地存储] 删除目录失败 {date_folder}: {e}")

            if deleted_count > 0:
                print(f"[本地存储] 共清理 {deleted_count} 个过期文件/目录")

            return deleted_count

        except Exception as e:
            print(f"[本地存储] 清理过期数据失败: {e}")
            return deleted_count

    def __del__(self):
        """析构函数，确保关闭连接"""
        self.cleanup()


class RollingWindowLocalBackend(LocalStorageBackend):
    """
    滚动窗口本地存储后端

    使用两个 SQLite 数据库文件实现热/冷数据分离：
    - current.db: 最近 N 天的热数据（频繁读写）
    - archive.db: 超过 N 天的归档数据（只读）

    优势：
    - 减少数据库文件数量（从 30+ 减少到 2 个）
    - 简化远程同步
    - 提高查询性能（热数据更小）
    """

    def __init__(
        self,
        data_dir: str = "output",
        enable_txt: bool = True,
        enable_html: bool = True,
        timezone: str = "Asia/Shanghai",
        hot_days: int = 7,
        archive_retention_days: int = 90,
    ):
        """
        初始化滚动窗口存储后端

        Args:
            data_dir: 数据目录路径
            enable_txt: 是否启用 TXT 快照
            enable_html: 是否启用 HTML 报告
            timezone: 时区配置
            hot_days: 热数据保留天数（在 current.db 中）
            archive_retention_days: 归档数据保留天数（0=永久保留）
        """
        super().__init__(data_dir, enable_txt, enable_html, timezone)
        self.hot_days = hot_days
        self.archive_retention_days = archive_retention_days

    @property
    def backend_name(self) -> str:
        return "rolling_window"

    def _get_db_path(self, date: Optional[str] = None, db_type: str = "news") -> Path:
        """
        获取 SQLite 数据库路径

        滚动窗口结构：output/{type}/current.db
        - output/news/current.db（热数据）
        - output/rss/current.db（热数据）

        Args:
            date: 日期字符串（在滚动窗口模式中忽略）
            db_type: 数据库类型 ("news" 或 "rss")

        Returns:
            数据库文件路径（始终返回 current.db）
        """
        db_dir = self.data_dir / db_type
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / "current.db"

    def _get_archive_db_path(self, db_type: str = "news") -> Path:
        """
        获取归档数据库路径

        Args:
            db_type: 数据库类型 ("news" 或 "rss")

        Returns:
            归档数据库路径
        """
        db_dir = self.data_dir / db_type
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / "archive.db"

    def _get_archive_connection(self, db_type: str = "news") -> sqlite3.Connection:
        """
        获取归档数据库连接

        Args:
            db_type: 数据库类型

        Returns:
            归档数据库连接
        """
        archive_path = str(self._get_archive_db_path(db_type))

        if archive_path not in self._db_connections:
            conn = sqlite3.connect(archive_path)
            conn.row_factory = sqlite3.Row
            self._init_tables(conn, db_type)
            self._db_connections[archive_path] = conn

        return self._db_connections[archive_path]

    def run_maintenance(self) -> Dict[str, int]:
        """
        运行维护任务：将过期的热数据迁移到归档数据库，并压缩归档HTML报告

        Returns:
            迁移统计 {"news_migrated": N, "rss_migrated": N, "archived_deleted": N, "html_archived": N}
        """
        stats = {"news_migrated": 0, "rss_migrated": 0, "archived_deleted": 0, "html_archived": 0}
        cutoff_date = (self._get_configured_time() - timedelta(days=self.hot_days)).strftime("%Y-%m-%d")

        for db_type in ["news", "rss"]:
            migrated = self._migrate_old_data(db_type, cutoff_date)
            stats[f"{db_type}_migrated"] = migrated

        # 清理过期归档数据
        if self.archive_retention_days > 0:
            stats["archived_deleted"] = self._cleanup_archive()

        # 压缩并归档旧的 HTML 报告
        stats["html_archived"] = self._archive_old_html_reports(cutoff_date)

        return stats

    def _migrate_old_data(self, db_type: str, cutoff_date: str) -> int:
        """
        将旧数据从 current.db 迁移到 archive.db

        Args:
            db_type: 数据库类型
            cutoff_date: 截止日期（早于此日期的数据将被迁移）

        Returns:
            迁移的记录数
        """
        current_conn = self._get_connection(db_type=db_type)
        archive_conn = self._get_archive_connection(db_type)

        migrated_count = 0

        try:
            # 获取需要迁移的数据
            if db_type == "news":
                cursor = current_conn.execute("""
                    SELECT * FROM news_items WHERE crawl_date < ?
                """, (cutoff_date,))
                rows = cursor.fetchall()

                for row in rows:
                    # 插入到归档数据库（忽略重复）
                    try:
                        archive_conn.execute("""
                            INSERT OR IGNORE INTO news_items
                            (title, platform_id, rank, url, mobile_url, crawl_date,
                             first_crawl_time, last_crawl_time, crawl_count, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (row["title"], row["platform_id"], row["rank"], row["url"],
                              row["mobile_url"], row["crawl_date"], row["first_crawl_time"],
                              row["last_crawl_time"], row["crawl_count"], row["created_at"],
                              row["updated_at"]))
                        migrated_count += 1
                    except sqlite3.IntegrityError:
                        pass  # 重复数据，跳过

                # 从 current.db 删除已迁移的数据
                if migrated_count > 0:
                    current_conn.execute("DELETE FROM news_items WHERE crawl_date < ?", (cutoff_date,))
                    current_conn.commit()
                    archive_conn.commit()

            elif db_type == "rss":
                cursor = current_conn.execute("""
                    SELECT * FROM rss_items WHERE crawl_date < ?
                """, (cutoff_date,))
                rows = cursor.fetchall()

                for row in rows:
                    try:
                        archive_conn.execute("""
                            INSERT OR IGNORE INTO rss_items
                            (feed_id, title, url, published_at, summary, author, crawl_date,
                             first_crawl_time, last_crawl_time, crawl_count, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (row["feed_id"], row["title"], row["url"], row["published_at"],
                              row["summary"], row["author"], row["crawl_date"],
                              row["first_crawl_time"], row["last_crawl_time"],
                              row["crawl_count"], row["created_at"], row["updated_at"]))
                        migrated_count += 1
                    except sqlite3.IntegrityError:
                        pass

                if migrated_count > 0:
                    current_conn.execute("DELETE FROM rss_items WHERE crawl_date < ?", (cutoff_date,))
                    current_conn.commit()
                    archive_conn.commit()

            if migrated_count > 0:
                print(f"[滚动窗口] 已迁移 {migrated_count} 条 {db_type} 数据到归档库")

        except Exception as e:
            print(f"[滚动窗口] 迁移 {db_type} 数据失败: {e}")

        return migrated_count

    def _cleanup_archive(self) -> int:
        """
        清理过期的归档数据

        Returns:
            删除的记录数
        """
        if self.archive_retention_days <= 0:
            return 0

        deleted_count = 0
        cutoff_date = (self._get_configured_time() - timedelta(days=self.archive_retention_days)).strftime("%Y-%m-%d")

        for db_type in ["news", "rss"]:
            try:
                archive_conn = self._get_archive_connection(db_type)
                table_name = "news_items" if db_type == "news" else "rss_items"

                cursor = archive_conn.execute(f"""
                    SELECT COUNT(*) FROM {table_name} WHERE crawl_date < ?
                """, (cutoff_date,))
                count = cursor.fetchone()[0]

                if count > 0:
                    archive_conn.execute(f"DELETE FROM {table_name} WHERE crawl_date < ?", (cutoff_date,))
                    archive_conn.commit()
                    deleted_count += count
                    print(f"[滚动窗口] 清理 {count} 条过期 {db_type} 归档数据")

            except Exception as e:
                print(f"[滚动窗口] 清理 {db_type} 归档失败: {e}")

        return deleted_count

    def _archive_old_html_reports(self, cutoff_date: str) -> int:
        """
        压缩并归档旧的 HTML 报告

        将早于 cutoff_date 的 HTML 报告打包为每日 tar.gz 归档，
        保存到 output/html/archive/YYYY/YYYY-MM-DD.tar.gz

        支持两种目录结构:
        - 新结构: output/html/YYYY-MM-DD/*.html
        - 旧结构: output/YYYY-MM-DD/*.html 或 output/YYYY-MM-DD/html/*.html

        Args:
            cutoff_date: 截止日期（YYYY-MM-DD 格式）

        Returns:
            归档的文件数
        """
        import tarfile

        archived_count = 0

        # 确保归档目录存在 (统一归档到 output/html/archive/)
        html_dir = self.data_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        archive_base = html_dir / "archive"

        # 收集每个日期的所有 HTML 文件
        date_files: dict[str, list[Path]] = {}

        def collect_html_files(folder: Path, date_str: str):
            """收集指定目录下的 HTML 文件"""
            if not folder.exists():
                return
            for html_file in folder.glob("*.html"):
                if date_str not in date_files:
                    date_files[date_str] = []
                date_files[date_str].append(html_file)

        def try_remove_empty_dir(folder: Path):
            """尝试删除空目录"""
            try:
                if folder.exists() and not any(folder.iterdir()):
                    folder.rmdir()
            except Exception:
                pass

        def is_old_date_folder(name: str) -> bool:
            """检查是否是需要归档的旧日期目录"""
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", name):
                return False
            return name < cutoff_date

        try:
            # 1. 新结构: output/html/YYYY-MM-DD/
            if html_dir.exists():
                for date_folder in html_dir.iterdir():
                    if not date_folder.is_dir():
                        continue
                    if date_folder.name in ("latest", "archive") or date_folder.name.startswith("."):
                        continue
                    if not is_old_date_folder(date_folder.name):
                        continue
                    collect_html_files(date_folder, date_folder.name)

            # 2. 旧结构: output/YYYY-MM-DD/ (日期目录在根目录)
            for date_folder in self.data_dir.iterdir():
                if not date_folder.is_dir():
                    continue
                if date_folder.name in ("html", "latest", "archive", "db") or date_folder.name.startswith("."):
                    continue
                if not is_old_date_folder(date_folder.name):
                    continue
                # 收集日期目录下的 HTML 文件
                collect_html_files(date_folder, date_folder.name)
                # 收集 html 子目录下的 HTML 文件
                html_subfolder = date_folder / "html"
                if html_subfolder.exists() and html_subfolder.is_dir():
                    collect_html_files(html_subfolder, date_folder.name)

            # 为每个日期创建 tar.gz 归档
            for date_str, html_files in date_files.items():
                if not html_files:
                    continue

                year = date_str[:4]
                archive_year_dir = archive_base / year
                archive_year_dir.mkdir(parents=True, exist_ok=True)

                # 归档文件名: YYYY-MM-DD.tar.gz
                tar_path = archive_year_dir / f"{date_str}.tar.gz"

                try:
                    with tarfile.open(tar_path, "w:gz") as tar:
                        for html_file in html_files:
                            # 使用原始文件名存入归档
                            tar.add(html_file, arcname=html_file.name)

                    # 归档成功后删除原文件
                    for html_file in html_files:
                        try:
                            html_file.unlink()
                            archived_count += 1
                        except Exception:
                            pass

                    # 清理空目录
                    for html_file in html_files:
                        try_remove_empty_dir(html_file.parent)
                        # 如果父目录的父目录也为空，也尝试删除
                        try_remove_empty_dir(html_file.parent.parent)

                    print(f"[滚动窗口] 已归档 {date_str}: {len(html_files)} 个文件 -> {tar_path.name}")

                except Exception as e:
                    print(f"[滚动窗口] 归档 {date_str} 失败: {e}")

            if archived_count > 0:
                print(f"[滚动窗口] 共归档 {archived_count} 个 HTML 报告")

        except Exception as e:
            print(f"[滚动窗口] HTML 归档失败: {e}")

        return archived_count

    def cleanup_old_data(self, retention_days: int) -> int:
        """
        清理过期数据（运行维护任务）

        在滚动窗口模式下，这会触发数据迁移和归档清理

        Args:
            retention_days: 保留天数（被 archive_retention_days 覆盖）

        Returns:
            处理的记录数
        """
        stats = self.run_maintenance()
        return stats["news_migrated"] + stats["rss_migrated"] + stats["archived_deleted"] + stats["html_archived"]
