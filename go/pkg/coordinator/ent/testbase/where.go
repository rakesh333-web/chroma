// Code generated by ent, DO NOT EDIT.

package testbase

import (
	"entgo.io/ent/dialect/sql"
	"github.com/chroma-core/chroma/go/pkg/coordinator/ent/predicate"
	"github.com/google/uuid"
)

// ID filters vertices based on their ID field.
func ID(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldID, id))
}

// IDEQ applies the EQ predicate on the ID field.
func IDEQ(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldID, id))
}

// IDNEQ applies the NEQ predicate on the ID field.
func IDNEQ(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldID, id))
}

// IDIn applies the In predicate on the ID field.
func IDIn(ids ...uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldID, ids...))
}

// IDNotIn applies the NotIn predicate on the ID field.
func IDNotIn(ids ...uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldID, ids...))
}

// IDGT applies the GT predicate on the ID field.
func IDGT(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldID, id))
}

// IDGTE applies the GTE predicate on the ID field.
func IDGTE(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldID, id))
}

// IDLT applies the LT predicate on the ID field.
func IDLT(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldID, id))
}

// IDLTE applies the LTE predicate on the ID field.
func IDLTE(id uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldID, id))
}

// ParentID applies equality check predicate on the "parent_id" field. It's identical to ParentIDEQ.
func ParentID(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldParentID, v))
}

// Name applies equality check predicate on the "name" field. It's identical to NameEQ.
func Name(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldName, v))
}

// CreatedAt applies equality check predicate on the "created_at" field. It's identical to CreatedAtEQ.
func CreatedAt(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldCreatedAt, v))
}

// UpdatedAt applies equality check predicate on the "updated_at" field. It's identical to UpdatedAtEQ.
func UpdatedAt(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldUpdatedAt, v))
}

// DeletedAt applies equality check predicate on the "deleted_at" field. It's identical to DeletedAtEQ.
func DeletedAt(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldDeletedAt, v))
}

// Version applies equality check predicate on the "version" field. It's identical to VersionEQ.
func Version(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldVersion, v))
}

// Text applies equality check predicate on the "text" field. It's identical to TextEQ.
func Text(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldText, v))
}

// ParentIDEQ applies the EQ predicate on the "parent_id" field.
func ParentIDEQ(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldParentID, v))
}

// ParentIDNEQ applies the NEQ predicate on the "parent_id" field.
func ParentIDNEQ(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldParentID, v))
}

// ParentIDIn applies the In predicate on the "parent_id" field.
func ParentIDIn(vs ...uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldParentID, vs...))
}

// ParentIDNotIn applies the NotIn predicate on the "parent_id" field.
func ParentIDNotIn(vs ...uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldParentID, vs...))
}

// ParentIDGT applies the GT predicate on the "parent_id" field.
func ParentIDGT(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldParentID, v))
}

// ParentIDGTE applies the GTE predicate on the "parent_id" field.
func ParentIDGTE(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldParentID, v))
}

// ParentIDLT applies the LT predicate on the "parent_id" field.
func ParentIDLT(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldParentID, v))
}

// ParentIDLTE applies the LTE predicate on the "parent_id" field.
func ParentIDLTE(v uuid.UUID) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldParentID, v))
}

// ParentIDIsNil applies the IsNil predicate on the "parent_id" field.
func ParentIDIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldParentID))
}

// ParentIDNotNil applies the NotNil predicate on the "parent_id" field.
func ParentIDNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldParentID))
}

// NameEQ applies the EQ predicate on the "name" field.
func NameEQ(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldName, v))
}

// NameNEQ applies the NEQ predicate on the "name" field.
func NameNEQ(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldName, v))
}

// NameIn applies the In predicate on the "name" field.
func NameIn(vs ...string) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldName, vs...))
}

// NameNotIn applies the NotIn predicate on the "name" field.
func NameNotIn(vs ...string) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldName, vs...))
}

// NameGT applies the GT predicate on the "name" field.
func NameGT(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldName, v))
}

// NameGTE applies the GTE predicate on the "name" field.
func NameGTE(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldName, v))
}

// NameLT applies the LT predicate on the "name" field.
func NameLT(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldName, v))
}

// NameLTE applies the LTE predicate on the "name" field.
func NameLTE(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldName, v))
}

// NameContains applies the Contains predicate on the "name" field.
func NameContains(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldContains(FieldName, v))
}

// NameHasPrefix applies the HasPrefix predicate on the "name" field.
func NameHasPrefix(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldHasPrefix(FieldName, v))
}

// NameHasSuffix applies the HasSuffix predicate on the "name" field.
func NameHasSuffix(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldHasSuffix(FieldName, v))
}

// NameIsNil applies the IsNil predicate on the "name" field.
func NameIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldName))
}

// NameNotNil applies the NotNil predicate on the "name" field.
func NameNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldName))
}

// NameEqualFold applies the EqualFold predicate on the "name" field.
func NameEqualFold(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldEqualFold(FieldName, v))
}

// NameContainsFold applies the ContainsFold predicate on the "name" field.
func NameContainsFold(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldContainsFold(FieldName, v))
}

// CreatedAtEQ applies the EQ predicate on the "created_at" field.
func CreatedAtEQ(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldCreatedAt, v))
}

// CreatedAtNEQ applies the NEQ predicate on the "created_at" field.
func CreatedAtNEQ(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldCreatedAt, v))
}

// CreatedAtIn applies the In predicate on the "created_at" field.
func CreatedAtIn(vs ...int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldCreatedAt, vs...))
}

// CreatedAtNotIn applies the NotIn predicate on the "created_at" field.
func CreatedAtNotIn(vs ...int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldCreatedAt, vs...))
}

// CreatedAtGT applies the GT predicate on the "created_at" field.
func CreatedAtGT(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldCreatedAt, v))
}

// CreatedAtGTE applies the GTE predicate on the "created_at" field.
func CreatedAtGTE(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldCreatedAt, v))
}

// CreatedAtLT applies the LT predicate on the "created_at" field.
func CreatedAtLT(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldCreatedAt, v))
}

// CreatedAtLTE applies the LTE predicate on the "created_at" field.
func CreatedAtLTE(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldCreatedAt, v))
}

// CreatedAtIsNil applies the IsNil predicate on the "created_at" field.
func CreatedAtIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldCreatedAt))
}

// CreatedAtNotNil applies the NotNil predicate on the "created_at" field.
func CreatedAtNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldCreatedAt))
}

// UpdatedAtEQ applies the EQ predicate on the "updated_at" field.
func UpdatedAtEQ(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldUpdatedAt, v))
}

// UpdatedAtNEQ applies the NEQ predicate on the "updated_at" field.
func UpdatedAtNEQ(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldUpdatedAt, v))
}

// UpdatedAtIn applies the In predicate on the "updated_at" field.
func UpdatedAtIn(vs ...int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldUpdatedAt, vs...))
}

// UpdatedAtNotIn applies the NotIn predicate on the "updated_at" field.
func UpdatedAtNotIn(vs ...int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldUpdatedAt, vs...))
}

// UpdatedAtGT applies the GT predicate on the "updated_at" field.
func UpdatedAtGT(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldUpdatedAt, v))
}

// UpdatedAtGTE applies the GTE predicate on the "updated_at" field.
func UpdatedAtGTE(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldUpdatedAt, v))
}

// UpdatedAtLT applies the LT predicate on the "updated_at" field.
func UpdatedAtLT(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldUpdatedAt, v))
}

// UpdatedAtLTE applies the LTE predicate on the "updated_at" field.
func UpdatedAtLTE(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldUpdatedAt, v))
}

// UpdatedAtIsNil applies the IsNil predicate on the "updated_at" field.
func UpdatedAtIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldUpdatedAt))
}

// UpdatedAtNotNil applies the NotNil predicate on the "updated_at" field.
func UpdatedAtNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldUpdatedAt))
}

// DeletedAtEQ applies the EQ predicate on the "deleted_at" field.
func DeletedAtEQ(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldDeletedAt, v))
}

// DeletedAtNEQ applies the NEQ predicate on the "deleted_at" field.
func DeletedAtNEQ(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldDeletedAt, v))
}

// DeletedAtIn applies the In predicate on the "deleted_at" field.
func DeletedAtIn(vs ...int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldDeletedAt, vs...))
}

// DeletedAtNotIn applies the NotIn predicate on the "deleted_at" field.
func DeletedAtNotIn(vs ...int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldDeletedAt, vs...))
}

// DeletedAtGT applies the GT predicate on the "deleted_at" field.
func DeletedAtGT(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldDeletedAt, v))
}

// DeletedAtGTE applies the GTE predicate on the "deleted_at" field.
func DeletedAtGTE(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldDeletedAt, v))
}

// DeletedAtLT applies the LT predicate on the "deleted_at" field.
func DeletedAtLT(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldDeletedAt, v))
}

// DeletedAtLTE applies the LTE predicate on the "deleted_at" field.
func DeletedAtLTE(v int64) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldDeletedAt, v))
}

// DeletedAtIsNil applies the IsNil predicate on the "deleted_at" field.
func DeletedAtIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldDeletedAt))
}

// DeletedAtNotNil applies the NotNil predicate on the "deleted_at" field.
func DeletedAtNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldDeletedAt))
}

// VersionEQ applies the EQ predicate on the "version" field.
func VersionEQ(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldVersion, v))
}

// VersionNEQ applies the NEQ predicate on the "version" field.
func VersionNEQ(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldVersion, v))
}

// VersionIn applies the In predicate on the "version" field.
func VersionIn(vs ...int) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldVersion, vs...))
}

// VersionNotIn applies the NotIn predicate on the "version" field.
func VersionNotIn(vs ...int) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldVersion, vs...))
}

// VersionGT applies the GT predicate on the "version" field.
func VersionGT(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldVersion, v))
}

// VersionGTE applies the GTE predicate on the "version" field.
func VersionGTE(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldVersion, v))
}

// VersionLT applies the LT predicate on the "version" field.
func VersionLT(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldVersion, v))
}

// VersionLTE applies the LTE predicate on the "version" field.
func VersionLTE(v int) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldVersion, v))
}

// VersionIsNil applies the IsNil predicate on the "version" field.
func VersionIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldVersion))
}

// VersionNotNil applies the NotNil predicate on the "version" field.
func VersionNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldVersion))
}

// TextEQ applies the EQ predicate on the "text" field.
func TextEQ(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldEQ(FieldText, v))
}

// TextNEQ applies the NEQ predicate on the "text" field.
func TextNEQ(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldNEQ(FieldText, v))
}

// TextIn applies the In predicate on the "text" field.
func TextIn(vs ...string) predicate.TestBase {
	return predicate.TestBase(sql.FieldIn(FieldText, vs...))
}

// TextNotIn applies the NotIn predicate on the "text" field.
func TextNotIn(vs ...string) predicate.TestBase {
	return predicate.TestBase(sql.FieldNotIn(FieldText, vs...))
}

// TextGT applies the GT predicate on the "text" field.
func TextGT(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldGT(FieldText, v))
}

// TextGTE applies the GTE predicate on the "text" field.
func TextGTE(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldGTE(FieldText, v))
}

// TextLT applies the LT predicate on the "text" field.
func TextLT(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldLT(FieldText, v))
}

// TextLTE applies the LTE predicate on the "text" field.
func TextLTE(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldLTE(FieldText, v))
}

// TextContains applies the Contains predicate on the "text" field.
func TextContains(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldContains(FieldText, v))
}

// TextHasPrefix applies the HasPrefix predicate on the "text" field.
func TextHasPrefix(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldHasPrefix(FieldText, v))
}

// TextHasSuffix applies the HasSuffix predicate on the "text" field.
func TextHasSuffix(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldHasSuffix(FieldText, v))
}

// TextIsNil applies the IsNil predicate on the "text" field.
func TextIsNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldIsNull(FieldText))
}

// TextNotNil applies the NotNil predicate on the "text" field.
func TextNotNil() predicate.TestBase {
	return predicate.TestBase(sql.FieldNotNull(FieldText))
}

// TextEqualFold applies the EqualFold predicate on the "text" field.
func TextEqualFold(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldEqualFold(FieldText, v))
}

// TextContainsFold applies the ContainsFold predicate on the "text" field.
func TextContainsFold(v string) predicate.TestBase {
	return predicate.TestBase(sql.FieldContainsFold(FieldText, v))
}

// And groups predicates with the AND operator between them.
func And(predicates ...predicate.TestBase) predicate.TestBase {
	return predicate.TestBase(sql.AndPredicates(predicates...))
}

// Or groups predicates with the OR operator between them.
func Or(predicates ...predicate.TestBase) predicate.TestBase {
	return predicate.TestBase(sql.OrPredicates(predicates...))
}

// Not applies the not operator on the given predicate.
func Not(p predicate.TestBase) predicate.TestBase {
	return predicate.TestBase(sql.NotPredicates(p))
}
