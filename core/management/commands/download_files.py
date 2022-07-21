from django.core.management.base import BaseCommand, CommandError
from searches import preprocess


class Command(BaseCommand):
    help = 'download data files'

    def handle(self, *args, **options):
        try:
            preprocess.download_foods()
            self.stdout.write(
                self.style.SUCCESS('data downloaded successfully')
            )

        except Exception as e:
            raise CommandError('download failed: , Error: "%s"' % e)
