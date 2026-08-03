import { Module } from '@nestjs/common';
import { PricingModule } from '../pricing/pricing.module';
import { JobsController } from './jobs.controller';
import { JobsService } from './jobs.service';

@Module({
  imports: [PricingModule],
  controllers: [JobsController],
  providers: [JobsService],
})
export class JobsModule {}
