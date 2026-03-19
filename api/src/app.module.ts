import { Module } from '@nestjs/common';
import { JobsModule } from './jobs/jobs.module';
import { LegalModule } from './legal/legal.module';

@Module({
  imports: [JobsModule, LegalModule],
})
export class AppModule {}
