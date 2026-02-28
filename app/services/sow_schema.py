from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExecutiveSummary(BaseModel):
	model_config = ConfigDict(extra="forbid")
	purpose: str
	methodology: str
	key_domains: List[str]
	art_of_possible: str


class Confidentiality(BaseModel):
	model_config = ConfigDict(extra="forbid")
	provisions: List[str]


class BusinessValue(BaseModel):
	model_config = ConfigDict(extra="forbid")
	value_opportunities: List[str]
	alignment_with_goals: List[str]


class Definitions(BaseModel):
	model_config = ConfigDict(extra="forbid", populate_by_name=True)
	Scope: str
	Deliverable: str
	Objectives: str
	Project_Resource: str = Field(
		validation_alias="Project Resource",
		serialization_alias="Project Resource",
	)
	Success_Criteria: str = Field(
		validation_alias="Success Criteria",
		serialization_alias="Success Criteria",
	)
	Exclusions: str


class ProjectResumptionCharge(BaseModel):
	model_config = ConfigDict(extra="forbid")
	description: str


class ScopeOfServices(BaseModel):
	model_config = ConfigDict(extra="forbid")
	activities: List[dict]
	deliverables_in_scope: List[str]
	out_of_scope: List[str]
	statement_of_understanding: List[str]
	assumptions: List[str]
	design_considerations: List[str]
	customer_responsibilities: List[str]
	project_resumption_charge: ProjectResumptionCharge


class ServiceDelivery(BaseModel):
	model_config = ConfigDict(extra="forbid")
	delivery_methodology: str


class RoleProfile(BaseModel):
	model_config = ConfigDict(extra="forbid")
	title: str
	responsibility: str


class ResourceProfiles(BaseModel):
	model_config = ConfigDict(extra="forbid")
	roles: List[RoleProfile]
	resource_allocation: str


class InvoicingItem(BaseModel):
	model_config = ConfigDict(extra="forbid")
	milestone: str
	payment_amount_usd: Optional[float] = None
	amount_usd: Optional[float] = None


class Fees(BaseModel):
	model_config = ConfigDict(extra="forbid")
	estimated_duration_weeks: int
	invoicing_schedule: List[InvoicingItem]
	notes: List[str]


class Expenses(BaseModel):
	model_config = ConfigDict(extra="forbid")
	reimbursed: List[str]


class ChangeManagement(BaseModel):
	model_config = ConfigDict(extra="forbid")
	requirement: str


class SignatureBlock(BaseModel):
	model_config = ConfigDict(extra="forbid")
	party: str
	entity: str
	fields: List[str]


class AcceptanceApprovals(BaseModel):
	model_config = ConfigDict(extra="forbid")
	terms: List[str]
	signature_blocks: List[SignatureBlock]


class Document(BaseModel):
	model_config = ConfigDict(extra="forbid")
	title: str
	date: str
	version: str
	provider: str
	client: str
	executive_summary: ExecutiveSummary
	confidentiality: Confidentiality
	business_value: BusinessValue
	definitions: Definitions
	scope_of_services: ScopeOfServices
	service_delivery: ServiceDelivery
	resource_profiles: ResourceProfiles
	fees: Fees
	expenses: Expenses
	change_management: ChangeManagement
	acceptance_and_approvals: AcceptanceApprovals


class SOWSchema(BaseModel):
	model_config = ConfigDict(extra="forbid")
	document: Document
