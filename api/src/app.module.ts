import { Module } from '@nestjs/common';
import { JobsModule } from './jobs/jobs.module';
import { LegalModule } from './legal/legal.module';
import { PricingModule } from './pricing/pricing.module';

@Module({
  imports: [JobsModule, LegalModule, PricingModule],
})
export class AppModule {}
